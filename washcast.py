"""Prototype hourly car-wash demand forecasting pipeline.

Offline: python washcast.py demo && python washcast.py run --traffic demo/traffic.csv --weather-csv demo/weather.csv --closures demo/closures.csv --forecast-start 2026-11-09
Online:  python washcast.py run --traffic real.csv --timezone America/Chicago
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pandas.tseries.holiday import USFederalHolidayCalendar
from urllib3.util.retry import Retry

TRAFFIC_COLS = ["timestamp", "location_id", "latitude", "longitude", "carwashes"]
WEATHER_COLS = ["timestamp", "location_id", "temperature_2m", "precipitation", "cloud_cover"]
CAT = ["location_id", "hour", "dow"]
NUM = ["latitude", "longitude", "month", "is_weekend", "is_holiday", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "temperature_2m", "precipitation", "cloud_cover", "rain_prev24h", "rain_prev72h"]
ALL_MODELS = ["baseline", "poisson", "negbin", "boosted", "mlp"]


def parse_timestamps(values: pd.Series, timezone: str | None, field: str) -> pd.Series:
    """Parse an entire column. Timezone-naive inputs require explicit timezone."""
    raw = values.astype('string').str.strip()
    aware = raw.str.contains(r'(?:Z|[+-]\d{2}:?\d{2})$', case=False, regex=True, na=False)
    if aware.any() and (~aware).any():
        raise ValueError(f"{field}: mixed timezone-aware and timezone-naive timestamps are not supported")
    if aware.any():
        result = pd.to_datetime(raw, utc=True, format='mixed', errors='coerce')
    else:
        if timezone is None:
            raise ValueError(f"{field}: timezone-naive timestamps need --timezone, e.g. America/Chicago")
        local = pd.to_datetime(raw, format='mixed', errors='coerce')
        result = local.dt.tz_localize(ZoneInfo(timezone), ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC')
    if result.isna().any():
        raise ValueError(f"{field}: {int(result.isna().sum())} invalid or DST-ambiguous timestamps. Export UTC/offset timestamps if possible.")
    return result


def load_traffic(path: str, timezone: str) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype={'location_id': 'string', 'location_code_id': 'string'})
    # Accept the car-wash SQL export columns without requiring a separate rename step.
    aliases = {'hour': 'timestamp', 'location_code_id': 'location_id',
               'num_carwashes': 'carwashes'}
    for old, new in aliases.items():
        if old in raw.columns:
            if new in raw.columns:
                raise ValueError(f'Both {old} and {new} are present in the traffic CSV; rename one')
            raw = raw.rename(columns={old: new})
    missing = sorted(set(TRAFFIC_COLS) - set(raw.columns))
    if missing:
        raise ValueError(f"Traffic file missing: {', '.join(missing)}; expected {TRAFFIC_COLS}")
    raw['timestamp'] = parse_timestamps(raw['timestamp'], timezone, 'traffic.timestamp').dt.floor('h')
    for col in ('latitude', 'longitude', 'carwashes'):
        raw[col] = pd.to_numeric(raw[col], errors='coerce')
    bad = raw[['latitude', 'longitude', 'carwashes', 'location_id']].isna().any(axis=1)
    bad |= (raw['carwashes'] < 0) | (raw['latitude'].abs() > 90) | (raw['longitude'].abs() > 180)
    if bad.any():
        raise ValueError(f"Traffic CSV has {int(bad.sum())} rows with invalid location, coordinates, or counts")
    if (raw['carwashes'] % 1 != 0).any():
        raise ValueError('carwashes must be nonnegative whole-number counts')
    raw['location_id'] = raw['location_id'].astype(str)
    # If one row is one wash, use carwashes=1. Multiple rows per hour are summed.
    hourly = raw.groupby(['location_id','timestamp'], as_index=False).agg(
        latitude=('latitude','median'), longitude=('longitude','median'), carwashes=('carwashes','sum'))
    for site, group in hourly.groupby('location_id'):
        if group['latitude'].max()-group['latitude'].min() > 0.02 or group['longitude'].max()-group['longitude'].min() > 0.02:
            raise ValueError(f'Site {site} has inconsistent coordinates; check export')
    return hourly.sort_values(['location_id','timestamp']).reset_index(drop=True)


def load_closures(path: str | None) -> set[tuple[str, date]]:
    if not path:
        return set()
    df = pd.read_csv(path, dtype={'location_id': 'string'})
    if not {'location_id','date'}.issubset(df.columns):
        raise ValueError('Closure CSV must contain location_id,date (local dates)')
    dates = pd.to_datetime(df['date'], errors='raise').dt.date
    return set(zip(df['location_id'].astype(str), dates))


def load_weather(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={'location_id': 'string'})
    missing = sorted(set(WEATHER_COLS)-set(df.columns))
    if missing:
        raise ValueError(f'Weather CSV missing columns: {missing}')
    df['timestamp'] = parse_timestamps(df['timestamp'], 'UTC', 'weather.timestamp').dt.floor('h')
    df['location_id'] = df['location_id'].astype(str)
    for c in WEATHER_COLS[2:]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df[WEATHER_COLS].drop_duplicates(['location_id','timestamp'], keep='last').sort_values(['location_id','timestamp']).reset_index(drop=True)


def json_get(session: requests.Session, url: str, params: dict, cache: Path) -> dict:
    cache.mkdir(parents=True,exist_ok=True)
    cache_key = hashlib.sha256((url + json.dumps(params,sort_keys=True)).encode()).hexdigest()
    file = cache / f'{cache_key}.json'
    if file.exists():
        return json.loads(file.read_text())
    resp = session.get(url,params=params,timeout=45)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get('error'):
        raise RuntimeError(f"Weather API: {payload.get('reason',payload)}")
    file.write_text(json.dumps(payload))
    return payload


def api_weather(traffic: pd.DataFrame, days_future: int, cache_dir: str) -> pd.DataFrame:
    """Historical near-real-time forecast analysis + live API, keyed by site and UTC hour.

    Historic inputs are reconstructed past conditions, NOT archived forecasts as of
    earlier decisions. Therefore backtest scores are diagnostic, not deployment-ready.
    """
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1.5, status_forcelist=[429,500,502,503,504])
    session.mount('https://', HTTPAdapter(max_retries=retry))
    api_key = os.environ.get('OPEN_METEO_API_KEY')
    past_url = os.environ.get('OPEN_METEO_HISTORY_URL', 'https://historical-forecast-api.open-meteo.com/v1/forecast')
    live_url = os.environ.get('OPEN_METEO_LIVE_URL', 'https://api.open-meteo.com/v1/forecast')
    now = pd.Timestamp.now(tz='UTC')
    oldest = (traffic.timestamp.min()-pd.Timedelta(days=4)).date()
    historical_end = min((now - pd.Timedelta(days=2)).date(), traffic.timestamp.max().date())
    results = []
    sites = traffic.groupby('location_id',as_index=False)[['latitude','longitude']].median()
    for _, s in sites.iterrows():
        site = s['location_id']
        params = {'latitude':round(float(s['latitude']),5),'longitude':round(float(s['longitude']),5),
                  'hourly':'temperature_2m,precipitation,cloud_cover', 'timezone':'UTC', 'timeformat':'unixtime'}
        if api_key:
            params['apikey'] = api_key
        frames = []
        if oldest <= historical_end:
            hist_params = {**params,'start_date':str(oldest),'end_date':str(historical_end)}
            payload = json_get(session,past_url,hist_params,Path(cache_dir))
            frames.append(parse_api_hourly(payload,site))
        live_params = {**params,'past_days':7,'forecast_days':min(16,max(9,days_future+2))}
        try:
            payload = json_get(session,live_url,live_params,Path(cache_dir))
            frames.append(parse_api_hourly(payload,site))
        except Exception as ex:
            print(f'Warning: live weather unavailable at site {site}: {ex}',file=sys.stderr)
        if not frames:
            raise RuntimeError(f'Weather unavailable for {site}')
        combined = pd.concat(frames,ignore_index=True).drop_duplicates(['location_id','timestamp'],keep='last')
        results.append(combined)
        print(f'Weather ready: site {site}, {len(combined)} hourly records')
    return pd.concat(results,ignore_index=True).sort_values(['location_id','timestamp']).reset_index(drop=True)


def parse_api_hourly(response: dict, site_id: str) -> pd.DataFrame:
    hour = response.get('hourly')
    if not hour or 'time' not in hour:
        raise ValueError(f'API returned no hourly weather for site {site_id}')
    result = pd.DataFrame({'timestamp':pd.to_datetime(hour['time'],unit='s',utc=True),'location_id':site_id})
    for key in WEATHER_COLS[2:]:
        result[key] = pd.to_numeric(pd.Series(hour.get(key,[np.nan]*len(result))),errors='coerce')
    return result


def weather_features(weather: pd.DataFrame) -> pd.DataFrame:
    # Reindex to real hourly grid, so a missing API hour is not treated as a 1-hour lag.
    all_sites = []
    for site, df in weather.groupby('location_id',sort=False):
        ts = pd.date_range(df.timestamp.min(),df.timestamp.max(),freq='h',tz='UTC')
        g = pd.DataFrame({'timestamp':ts, 'location_id':site}).merge(df,on=['location_id','timestamp'],how='left')
        for hrs, min_obs in [(24,18),(72,54)]:
            lagged = g['precipitation'].shift(1)
            g[f'rain_prev{hrs}h'] = lagged.rolling(hrs,min_periods=min_obs).sum()
        all_sites.append(g)
    return pd.concat(all_sites,ignore_index=True)


def add_calendar_features(df: pd.DataFrame, timezone: str) -> pd.DataFrame:
    df = df.copy()
    local = df.timestamp.dt.tz_convert(timezone)
    df['local_date'] = local.dt.date
    df['hour'] = local.dt.hour.astype(int)
    df['dow'] = local.dt.dayofweek.astype(int)
    df['month'] = local.dt.month.astype(int)
    df['is_weekend'] = (df['dow']>=5).astype(int)
    holidays = set(USFederalHolidayCalendar().holidays(
        start=(local.min()-pd.Timedelta(days=2)).date(),
        end=(local.max()+pd.Timedelta(days=2)).date()).date)
    df['is_holiday'] = df['local_date'].isin(holidays).astype(int)
    df['hour_sin'] = np.sin(2*np.pi*df.hour/24)
    df['hour_cos'] = np.cos(2*np.pi*df.hour/24)
    df['dow_sin'] = np.sin(2*np.pi*df.dow/7)
    df['dow_cos'] = np.cos(2*np.pi*df.dow/7)
    return df


def prepare(traffic: pd.DataFrame, weather: pd.DataFrame, timezone: str) -> pd.DataFrame:
    weather = weather_features(weather)
    joined = traffic.merge(weather,on=['location_id','timestamp'],how='left',validate='one_to_one')
    return add_calendar_features(joined,timezone)


def open_hour_mask(local: pd.Series, open_hour: int = 8,
                   close_hour: int = 19, sunday_open_hour: int | None = None) -> pd.Series:
    """Scheduled timestamp starts: Monday-Saturday 08:00, Sunday 09:00, close 19:00.

    If sunday_open_hour is None, use the weekday opening time (compatibility
    for older helper calls). The 19:00 hour is not included.
    """
    sunday_start = open_hour if sunday_open_hour is None else sunday_open_hour
    opens = np.where(local.dt.dayofweek == 6, sunday_start, open_hour)
    return (local.dt.hour >= opens) & (local.dt.hour < close_hour)


def fill_missing_open_hours(traffic: pd.DataFrame, closures: set, timezone: str,
                            open_hour: int = 8, close_hour: int = 19,
                            sunday_open_hour: int = 9) -> tuple[pd.DataFrame, dict]:
    """Infer zero ticket counts ONLY on store-days with at least one observed hour.

    An entirely unobserved local store-day is unknown/closed, not a trustworthy
    zero-demand day. Remove it from labeled data (training and evaluation), while
    preserving its date and excluded-hour count in the quality report. Recorded
    zero counts are always preserved. Do not extrapolate beyond a site's first
    and last recorded timestamp; honor separately specified known closures.

    A partially observed day still assumes that missing ticket-count hours mean
    zero. This assumption should be reviewed against outage/closure records.
    """
    if traffic.empty:
        raise ValueError('Cannot generate operating hours from an empty traffic table')
    pieces = []
    summary = {}
    for site, g in traffic.groupby('location_id', sort=False):
        first, last = g.timestamp.min(), g.timestamp.max()
        # UTC grid handles DST transitions, including duplicate local fall-back hours.
        all_hours = pd.Series(pd.date_range(first, last, freq='h', tz='UTC'))
        local = all_hours.dt.tz_convert(timezone)
        keep = open_hour_mask(local, open_hour, close_hour, sunday_open_hour)
        keep &= ~pd.Series([(str(site), d) in closures for d in local.dt.date])
        hours = pd.DatetimeIndex(all_hours.loc[keep].tolist())
        frame = pd.DataFrame({'location_id': str(site), 'timestamp': hours})
        frame = frame.merge(g[['timestamp', 'carwashes']], on='timestamp', how='left',
                            validate='one_to_one')
        frame['inferred_zero'] = frame.carwashes.isna()
        local_dates = frame.timestamp.dt.tz_convert(timezone).dt.date
        # Group by LOCAL date: the same operating day must not become a different
        # day after conversion to UTC (including daylight-saving transitions).
        fully_missing = frame.groupby(local_dates)['inferred_zero'].transform('all')
        full_days = sorted(str(x) for x in local_dates.loc[fully_missing].unique())
        excluded_hours = int(fully_missing.sum())
        raw_missing_hours = int(frame.inferred_zero.sum())
        frame = frame.loc[~fully_missing].copy()
        frame['carwashes'] = frame['carwashes'].fillna(0).astype('int64')
        frame['latitude'] = float(g.latitude.median())
        frame['longitude'] = float(g.longitude.median())
        summary[str(site)] = {
            'observed_operating_hours': int((~frame.inferred_zero).sum()),
            'inferred_zero_hours': int(frame.inferred_zero.sum()),
            'raw_missing_operating_hours': raw_missing_hours,
            'fully_inferred_zero_dates': full_days,  # historical report key
            'excluded_full_missing_dates': full_days,
            'excluded_full_missing_days': len(full_days),
            'excluded_full_missing_hours': excluded_hours,
            'first_observed_hour_utc': str(first),
            'last_observed_hour_utc': str(last),
        }
        pieces.append(frame)
    filled = pd.concat(pieces, ignore_index=True)
    filled = filled[['timestamp','location_id','latitude','longitude','carwashes','inferred_zero']]
    filled = filled.sort_values(['location_id','timestamp']).reset_index(drop=True)
    return filled, {
        'per_site': summary,
        'total_inferred_zero_hours': int(filled.inferred_zero.sum()),
        'total_excluded_full_missing_days': sum(x['excluded_full_missing_days'] for x in summary.values()),
        'total_excluded_full_missing_hours': sum(x['excluded_full_missing_hours'] for x in summary.values()),
        'warning': ('Fully missing store-days are excluded, never converted to zero labels. '
                    'Missing hours on partly observed days are inferred zeros, not verified counts; '
                    'review outages and partial-day closures.'),
    }


def quality_report(traffic: pd.DataFrame, closures: set, timezone: str,
                   open_hour: int, close_hour: int,
                   sunday_open_hour: int | None = None) -> dict:
    sites = {}
    for site, g in traffic.groupby('location_id'):
        local = g.timestamp.dt.tz_convert(timezone)
        first = local.min().normalize()
        last = local.max().normalize()+pd.DateOffset(days=1)
        grid = pd.date_range(start=first,end=last,freq='h',inclusive='left')
        scheduled = grid[open_hour_mask(pd.Series(grid), open_hour, close_hour,
                                        sunday_open_hour).to_numpy()]
        expected = pd.DatetimeIndex([ts for ts in scheduled if (site,ts.date()) not in closures]).tz_convert('UTC')
        observed = pd.DatetimeIndex(g.timestamp)
        missing = expected.difference(observed)
        sites[site] = {'observed_rows':int(len(g)), 'expected_nominal_open_hours':int(len(expected)),
                       'missing_nominal_open_hours':int(len(missing)),
                       'missing_pct':round(100*len(missing)/max(1,len(expected)),2),
                       'first_observed':str(g.timestamp.min()),'last_observed':str(g.timestamp.max()),
                       'inferred_zero_hours':int(g.inferred_zero.sum()) if 'inferred_zero' in g else 0}
    return {'sites':sites,'disclaimer':'Nominal operating hours are assumptions; inferred zeros require validation against outages and closures.'}


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    x = df[CAT+NUM].copy()
    for c in CAT:
        x[c] = x[c].astype(str)
    return x


def transformer() -> ColumnTransformer:
    return ColumnTransformer([
        ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CAT),
        ('num',Pipeline([('fill',SimpleImputer(strategy='median',add_indicator=False)),
                         ('scale',StandardScaler())]),NUM),
    ],remainder='drop',sparse_threshold=0)


class Baseline:
    """Site, weekday and hour, progressively falling back to less granular medians."""
    def fit(self, df: pd.DataFrame):
        self.exact = df.groupby(['location_id','dow','hour'])['carwashes'].median()
        self.site_hour = df.groupby(['location_id','hour'])['carwashes'].median()
        self.site = df.groupby('location_id')['carwashes'].median()
        self.global_median = df.carwashes.median()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([float(self.exact.get((r.location_id,r.dow,r.hour),
                                self.site_hour.get((r.location_id,r.hour),
                                self.site.get(r.location_id,self.global_median))))
            for r in df.itertuples(index=False)])


class NegativeBinomial:
    """Approximate NB GLM with dispersion estimated from a Poisson fit's residuals."""
    def fit(self, x: pd.DataFrame, y: np.ndarray):
        import statsmodels.api as sm
        self.trans = transformer()
        a = self.trans.fit_transform(x)
        mu = np.maximum(np.asarray(PoissonRegressor(alpha=0.3,max_iter=250).fit(a,y).predict(a)),1)
        alpha = np.mean(((y-mu)**2-mu)/np.maximum(mu**2,1))
        self.dispersion = float(np.clip(alpha,0.05,2.5))
        design = sm.add_constant(a,has_constant='add')
        # Regularization helps with sparse site x hour / holiday categories.
        model = sm.GLM(y,design,family=sm.families.NegativeBinomial(alpha=self.dispersion))
        self.params = np.asarray(model.fit_regularized(alpha=0.0005,L1_wt=0,maxiter=150).params)
        return self

    def predict(self,x: pd.DataFrame) -> np.ndarray:
        import statsmodels.api as sm
        a = self.trans.transform(x)
        return np.exp(np.clip(sm.add_constant(a,has_constant='add')@self.params,-10,12))


def make_model(name: str):
    if name=='poisson':
        return Pipeline([('features',transformer()),('model',PoissonRegressor(alpha=0.15,max_iter=300))])
    if name=='boosted':
        return Pipeline([('features',transformer()),('model',HistGradientBoostingRegressor(
            loss='poisson',max_iter=130,max_leaf_nodes=19,learning_rate=0.07,l2_regularization=0.5,random_state=42))])
    if name=='mlp':
        return Pipeline([('features',transformer()),('model',MLPRegressor(
            hidden_layer_sizes=(24,12),solver='adam',alpha=1.5,max_iter=120,
            early_stopping=False,n_iter_no_change=12,random_state=42))])
    if name=='negbin':
        return NegativeBinomial()
    raise ValueError(f'Unknown model {name}')


def predict_model(name: str,model,df: pd.DataFrame) -> np.ndarray:
    if name=='baseline':
        return np.maximum(0,model.predict(df))
    return np.maximum(0,model.predict(feature_matrix(df)))


def fit_model(name: str,df: pd.DataFrame):
    if name=='baseline':
        return Baseline().fit(df)
    model = make_model(name)
    model.fit(feature_matrix(df),df.carwashes.to_numpy(dtype=float))
    return model


def model_metrics(true: np.ndarray,pred: np.ndarray) -> dict:
    true,pred = np.asarray(true),np.asarray(pred)
    return {'mae':round(float(mean_absolute_error(true,pred)),3),
            'rmse':round(float(np.sqrt(mean_squared_error(true,pred))),3),
            'wape_percent':round(100*float(np.sum(abs(true-pred))/max(1,np.sum(true))),2),
            'observations':int(len(true))}


def future_hours(start_date: date, days: int, timezone: str, open_hour: int,
                 close_hour: int, sunday_open_hour: int | None = None) -> pd.DatetimeIndex:
    start = pd.Timestamp(start_date).tz_localize(ZoneInfo(timezone))
    end = start + pd.DateOffset(days=days)
    local = pd.date_range(start,end,freq='h',inclusive='left')
    schedule = open_hour_mask(pd.Series(local), open_hour, close_hour, sunday_open_hour)
    return local[schedule.to_numpy()].tz_convert('UTC')


def build_future(traffic: pd.DataFrame, weather: pd.DataFrame, forecast_start: date,
                 forecast_days: int, timezone: str, open_hour: int, close_hour: int,
                 closures: set, sunday_open_hour: int | None = None):
    hours = future_hours(forecast_start,forecast_days,timezone,open_hour,close_hour,
                         sunday_open_hour)
    sites = traffic.groupby('location_id',as_index=False)[['latitude','longitude']].median()
    pieces=[]
    for _,s in sites.iterrows():
        one=pd.DataFrame({'timestamp':hours,'location_id':s.location_id,
                          'latitude':s.latitude,'longitude':s.longitude})
        pieces.append(one)
    future=pd.concat(pieces,ignore_index=True)
    future=add_calendar_features(future,timezone)
    future=future[~pd.Series(list(zip(future.location_id,future.local_date)),index=future.index).isin(closures)].copy()
    weather=weather_features(weather)
    future=future.merge(weather,on=['location_id','timestamp'],how='left',validate='one_to_one')
    missing=int(future['temperature_2m'].isna().sum())
    if missing:
        raise RuntimeError(f'{missing} forecast location-hours lack weather. Extend forecast coverage, shorten horizon, or check the API.')
    return future


def run(args) -> None:
    traffic=load_traffic(args.traffic,args.timezone)
    closures=load_closures(args.closures)
    full_count=len(traffic)
    max_ts=traffic.timestamp.max()
    traffic=traffic[traffic.timestamp >= max_ts - pd.Timedelta(days=args.train_days)].copy()
    local=traffic.timestamp.dt.tz_convert(args.timezone)
    traffic=traffic[open_hour_mask(local,args.open_hour,args.close_hour,
                                   args.sunday_open_hour)].copy()
    local=traffic.timestamp.dt.tz_convert(args.timezone)
    known_closed=pd.Series(list(zip(traffic.location_id,local.dt.date)),index=traffic.index).isin(closures)
    traffic=traffic[~known_closed].copy()
    if traffic.empty:
        raise ValueError('No usable traffic rows after filtering')
    prefill_report=quality_report(traffic,closures,args.timezone,args.open_hour,
                                  args.close_hour,args.sunday_open_hour)
    traffic,fill_report=fill_missing_open_hours(traffic,closures,args.timezone,
                           args.open_hour,args.close_hour,args.sunday_open_hour)
    print(f'Loaded {len(traffic):,} hourly observations, {traffic.location_id.nunique()} sites; '
          f'{fill_report["total_inferred_zero_hours"]:,} zero-count hours inferred on active days; '
          f'{fill_report["total_excluded_full_missing_days"]:,} full missing store-days excluded.')
    report={'before_fill':prefill_report,'zero_fill':fill_report}
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    (out/'data_quality.json').write_text(json.dumps(report,indent=2))
    print('Data quality: fully missing store-days excluded, isolated missing hours inferred as zero; see data_quality.json.')
    if args.weather_csv:
        weather=load_weather(args.weather_csv)
        source='supplied offline weather CSV'
    else:
        weather=api_weather(traffic,args.forecast_days,args.cache_dir)
        source='Open-Meteo historical analysis + live forecast'
    df=prepare(traffic,weather,args.timezone)
    # Inspectable audit trail of every labeled traffic hour and its joined weather/features.
    df.to_csv(out/'merged_training_data.csv',index=False)
    frac_missing=float(df.temperature_2m.isna().mean())
    if frac_missing>0.05:
        warnings.warn(f'{frac_missing:.1%} of labeled hours do not have matching weather; models impute missing weather')
    days=sorted(set(df.local_date))
    if len(days)<args.validation_days+14:
        raise ValueError(f'Need >= {args.validation_days+14} calendar days of traffic; got {len(days)}')
    cutoff=days[-args.validation_days]
    train=df[df.local_date<cutoff].copy()
    val=df[df.local_date>=cutoff].copy()
    if len(train)<100 or len(val)<30:
        raise ValueError('Too few training or validation observations for this split')
    selected_models=[v.strip() for v in args.models.split(',') if v.strip()]
    if any(name not in ALL_MODELS for name in selected_models):
        raise ValueError(f'--models must be chosen from {ALL_MODELS}')
    results=[]
    all_predictions=val[['location_id','timestamp','local_date','hour','carwashes']].copy()
    calibrated={}
    for name in selected_models:
        print(f'Fitting {name} on {len(train):,} rows ...',flush=True)
        try:
            model=fit_model(name,train)
            pred=predict_model(name,model,val)
            result={'model':name,**model_metrics(val.carwashes.to_numpy(),pred)}
            results.append(result)
            all_predictions[f'pred_{name}']=np.round(pred,2)
            # Diagnostic multiplicative count-scale band from absolute validation errors.
            scores=np.abs(val.carwashes.to_numpy()-pred)/np.sqrt(np.maximum(pred,1))
            calibrated[name]=float(np.quantile(scores,0.8))
            print(f'  MAE {result["mae"]}, RMSE {result["rmse"]}, WAPE {result["wape_percent"]}%')
        except Exception as exc:
            print(f'  Model {name} failed: {exc}',file=sys.stderr)
    if not results:
        raise RuntimeError('All models failed')
    compare=pd.DataFrame(results).sort_values('mae').reset_index(drop=True)
    compare.to_csv(out/'model_comparison.csv',index=False)
    all_predictions.to_csv(out/'validation_predictions.csv',index=False)
    winner=compare.iloc[0]['model']
    start=date.fromisoformat(args.forecast_start) if args.forecast_start else pd.Timestamp.now(tz=args.timezone).date()+timedelta(days=1)
    future=build_future(traffic,weather,start,args.forecast_days,args.timezone,args.open_hour,args.close_hour,closures,args.sunday_open_hour)
    # Refit the best model on all labeled hours before making the forward forecast.
    print(f'Refitting best model ({winner}) on all {len(df):,} rows ...')
    best_model=fit_model(winner,df)
    yhat=predict_model(winner,best_model,future)
    halfwidth=calibrated[winner]*np.sqrt(np.maximum(yhat,1))
    future['expected_washes']=np.round(yhat,1)
    future['lower_80_approx']=np.round(np.maximum(0,yhat-halfwidth),1)
    future['upper_80_approx']=np.round(yhat+halfwidth,1)
    future['local_time']=future.timestamp.dt.tz_convert(args.timezone).astype(str)
    keep=['local_time','timestamp','location_id','expected_washes','lower_80_approx','upper_80_approx',
          'temperature_2m','precipitation','rain_prev24h','rain_prev72h']
    future[keep].to_csv(out/'hourly_forecast.csv',index=False)
    daily=future.groupby(['location_id','local_date'],as_index=False).agg(
        expected_washes=('expected_washes','sum'),
        lower_80_approx=('lower_80_approx','sum'),upper_80_approx=('upper_80_approx','sum'))
    daily.to_csv(out/'daily_forecast.csv',index=False)
    joblib.dump(best_model,out/'best_model.joblib')
    meta={'model':winner,'weather_source':source,'site_count':int(df.location_id.nunique()),
          'usable_labeled_hours':int(len(df)),'training_window_days':args.train_days,
          'validation_start_local':str(cutoff),'forecast_start_local':str(start),
          'timezone':args.timezone,'open_hour':args.open_hour,'sunday_open_hour':args.sunday_open_hour,'close_hour':args.close_hour,
          'weather_observation_leakage_warning': 'Historical features are realized/reconstructed weather, not archived as-of forecast runs. Backtest scores are optimistic for real next-day / 7-day decisions.',
          'interval_warning':'Illustrative 80% validation-residual bands, not independently calibrated and not guaranteed conditional coverage; summing hourly bands is not a calibrated daily interval.',
          'forecast_weather_gap_hours':int(future.temperature_2m.isna().sum()),
          'missing_weather_fraction_on_training':round(frac_missing,5)}
    (out/'run_summary.json').write_text(json.dumps(meta,indent=2))
    print(f'\nFinished. Best validation MAE: {winner} ({compare.iloc[0]["mae"]} washes/hour).')
    print(f'Outputs: {out.resolve()}')
    print('CAUTION: Scores use observed historical weather; intervals are exploratory, not production-calibrated.')


def generate_demo(args) -> None:
    """Generate a completely synthetic, weather-dependent 16-site car-wash dataset."""
    rng=np.random.default_rng(args.seed)
    directory=Path(args.output)
    directory.mkdir(parents=True,exist_ok=True)
    tz=ZoneInfo(args.timezone)
    last=date.fromisoformat(args.end_date)
    first=last-timedelta(days=args.days-1)
    start=pd.Timestamp(first).tz_localize(tz)
    finish=pd.Timestamp(last+timedelta(days=1)+timedelta(days=args.future_days)).tz_localize(tz)
    local_time=pd.date_range(start,finish,freq='h',inclusive='left')
    hours=local_time.tz_convert('UTC')
    weather_rows=[]; traffic_rows=[]; closure_rows=[]
    dt_days=np.asarray([(d-first).days for d in local_time.date])
    dow=np.asarray(local_time.dayofweek); hour=np.asarray(local_time.hour)
    for i in range(args.sites):
        site=f'{100+i}'
        latitude=33.1 + 0.16*(i%8) + rng.normal(0,.015)
        longitude=-88.5 + 0.22*(i//8)+rng.normal(0,.02)
        # Rain fronts evolve across days, with persistent wet spells and local variability.
        wet=np.zeros(args.days+args.future_days+1)
        for d in range(len(wet)):
            wet[d]= (0.73*wet[d-1] if d else 0) + rng.normal(0,.65)
        rain_probability=1/(1+np.exp(-(wet[dt_days]-.4)*2))
        rain_indicator=rng.random(len(hours)) < (rain_probability * .28)
        precipitation=rain_indicator*rng.gamma(shape=1.6,scale=1.7,size=len(hours))
        temp=20+5*np.sin(2*np.pi*(hour-8)/24)+3*np.sin(2*np.pi*dt_days/70)+rng.normal(0,1.8,len(hours))
        cloud=np.clip(27+65*rain_probability+rng.normal(0,12,len(hours)),0,100)
        rain_prior=pd.Series(precipitation).shift(1).rolling(72,min_periods=54).sum().fillna(0).to_numpy()
        base=rng.uniform(16,38)
        hour_shape=np.where((hour>=7)&(hour<19), .4 + 1.2*np.exp(-((hour-13)/4.2)**2),0)
        weekday=np.where(dow==5,1.25,np.where(dow==6,.88,1.0))
        weather_effect=np.exp(-.29*precipitation + .11*np.minimum(rain_prior/12,2))
        temp_effect=np.clip(1+.014*(temp-22),.7,1.3)
        expected=base*hour_shape*weekday*weather_effect*temp_effect
        counts=rng.poisson(expected)
        closure_dates=set()
        for d in range(args.days):
            if rng.random()<.015:
                day=first+timedelta(days=d)
                closure_dates.add(day)
                if rng.random()<.7:
                    closure_rows.append((site,day.isoformat()))
        for j,utc in enumerate(hours):
            weather_rows.append((utc.isoformat(),site,round(float(temp[j]),2),round(float(precipitation[j]),3),round(float(cloud[j]),1)))
            if local_time[j].date()>last or not 7<=hour[j]<19:
                continue
            if local_time[j].date() in closure_dates or rng.random()<.025:
                continue
            traffic_rows.append((utc.isoformat(),site,round(latitude,5),round(longitude,5),int(counts[j])))
    traffic_out=directory/'traffic.csv'
    weather_out=directory/'weather.csv'
    closes_out=directory/'closures.csv'
    pd.DataFrame(traffic_rows,columns=TRAFFIC_COLS).to_csv(traffic_out,index=False)
    pd.DataFrame(weather_rows,columns=WEATHER_COLS).to_csv(weather_out,index=False)
    pd.DataFrame(closure_rows,columns=['location_id','date']).to_csv(closes_out,index=False)
    print(f'Demo generated: {traffic_out}, {weather_out}, {closes_out}')
    print(f'{args.sites} fictitious sites, {args.days} historical days, {args.future_days} future days, closures + intentionally missing hourly rows.')


def create_parser():
    ap=argparse.ArgumentParser(description='Car-wash traffic forecast prototype')
    sub=ap.add_subparsers(dest='command',required=True)
    d=sub.add_parser('demo',help='generate synthetic weather-dependent traffic and weather')
    d.add_argument('--output',default='demo')
    d.add_argument('--sites',type=int,default=16)
    d.add_argument('--days',type=int,default=96)
    d.add_argument('--future-days',type=int,default=8)
    d.add_argument('--end-date',default='2026-11-08',help='last synthetic historical traffic date, YYYY-MM-DD')
    d.add_argument('--timezone',default='America/Chicago')
    d.add_argument('--seed',type=int,default=42)
    r=sub.add_parser('run',help='train, validate models, and forecast hourly traffic')
    r.add_argument('--traffic',required=True,help='CSV: timestamp,location_id,latitude,longitude,carwashes')
    r.add_argument('--weather-csv',help='offline combined observed + future weather CSV; omitted = API')
    r.add_argument('--closures',help='optional site closure calendar: location_id,date')
    r.add_argument('--timezone',default='America/Chicago',help='timezone for naive input and local operating hours')
    r.add_argument('--open-hour',type=int,default=8,help='Monday-Saturday opening hour (default 8)')
    r.add_argument('--sunday-open-hour',type=int,default=9,help='Sunday opening hour (default 9)')
    r.add_argument('--close-hour',type=int,default=19)
    r.add_argument('--train-days',type=int,default=90)
    r.add_argument('--validation-days',type=int,default=14)
    r.add_argument('--forecast-days',type=int,default=7)
    r.add_argument('--forecast-start',help='local date YYYY-MM-DD; defaults to tomorrow')
    r.add_argument('--models',default='baseline,poisson,negbin,boosted,mlp')
    r.add_argument('--output',default='results')
    r.add_argument('--cache-dir',default='.weather_cache')
    return ap


def main(argv=None):
    ap=create_parser()
    args=ap.parse_args(argv)
    if args.command=='demo':
        if args.days<35 or args.sites<1:
            ap.error('demo needs >=35 days and >=1 site')
        generate_demo(args)
    elif args.command=='run':
        if not 0<=args.open_hour<args.close_hour<=24 or not 0<=args.sunday_open_hour<args.close_hour:
            ap.error('Operating hours must satisfy 0 <= both opening hours < closing hour <= 24')
        if not 1<=args.forecast_days<=7:
            ap.error('This prototype supports 1-7 future days')
        run(args)


if __name__=='__main__':
    main()
