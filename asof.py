"""Forecast-origin-aware training/backtesting and reusable forward predictions.

Crucial separation: labels = observed washes; target-hour inputs = weather that was
available at the forecast *issuance*. Never substitute realized target weather in
historical as-of rows. The only remaining approximation is reconstructed weather
for hours preceding each issuance (not a frozen historical observation feed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
import sklearn
from sklearn.exceptions import InconsistentVersionWarning
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import washcast as core

# One model can learn that weather is less certain at longer forecast horizons.
if 'lead_day' not in core.NUM:
    core.NUM.append('lead_day')

ARCHIVE_URL = 'https://single-runs-api.open-meteo.com/v1/forecast'
MODEL_ID = 'ecmwf_ifs'
HOURS = 'temperature_2m,precipitation,cloud_cover'
DATA_START = '2024-03-14'  # documented beginning of ECMWF HRES single-run archive
KEY_COLS = ['issued_at', 'location_id', 'timestamp']


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def runtime_versions() -> dict:
    return {'python': sys.version.split()[0], 'scikit_learn': sklearn.__version__,
            'numpy': np.__version__, 'pandas': pd.__version__, 'joblib': joblib.__version__}


def labeled_keys(rows: pd.DataFrame) -> pd.DataFrame:
    keys = rows[KEY_COLS].copy()
    keys['issued_at'] = pd.to_datetime(keys.issued_at, utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    keys['timestamp'] = pd.to_datetime(keys.timestamp, utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    keys['location_id'] = keys.location_id.astype(str)
    return keys.drop_duplicates().sort_values(KEY_COLS).reset_index(drop=True)


def saved_labeled_keys(folder: Path) -> pd.DataFrame:
    path = folder/'model_training_keys.csv'
    if path.exists():
        old = pd.read_csv(path, dtype={'location_id': str})
        return labeled_keys(old)
    # Backwards compatibility: v0.2 stored the historical labeled examples.
    old_path = folder/'asof_training_examples.csv'
    if old_path.exists():
        return labeled_keys(pd.read_csv(old_path, usecols=KEY_COLS, dtype={'location_id': str}))
    raise ValueError('Missing training-key ledger and historical training examples. Do a full refit before incremental updates.')


def load_saved_model(folder: Path, name: str, meta: dict):
    original = meta.get('runtime_versions', {}).get('scikit_learn')
    if original and original != sklearn.__version__:
        raise RuntimeError(
            f'Saved {name} model uses scikit-learn {original}, but this environment uses {sklearn.__version__}. '
            'Loading across versions is unsupported. Refit in this environment (asof.py fit), '
            'or recreate the original environment exactly. Do not suppress the warning.'
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', InconsistentVersionWarning)
            return joblib.load(folder/'models'/f'{name}.joblib')
    except (InconsistentVersionWarning, ModuleNotFoundError, AttributeError, ImportError) as exc:
        raise RuntimeError(
            f'Cannot load {name} model in this environment ({sklearn.__version__}). '
            'The saved model is probably from another scikit-learn/Python version. '
            'Do a fresh fit locally with the same dependency versions you will use for forecasts.'
        ) from exc



def issue_time(local_date: date, tz: str, hour: int = 18) -> pd.Timestamp:
    """When a manager would issue a forecast, local time (default 6pm)."""
    return pd.Timestamp(local_date).tz_localize(ZoneInfo(tz)) + pd.Timedelta(hours=hour)


def safely_available_run(issue_utc: pd.Timestamp) -> pd.Timestamp:
    """Latest 12 UTC run initialized >=8 hours before decision (conservative)."""
    guard = issue_utc.tz_convert('UTC') - pd.Timedelta(hours=8)
    candidate = guard.normalize() + pd.Timedelta(hours=12)
    if candidate > guard:
        candidate -= pd.Timedelta(days=1)
    return candidate


def session_with_retries() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504])
    s.mount('https://', HTTPAdapter(max_retries=retry))
    return s


def load_archive_csv(path: str) -> pd.DataFrame:
    """One forecast vintage per (issued_at, location_id, target timestamp)."""
    df = pd.read_csv(path, dtype={'location_id': 'string'})
    required = {'issued_at', *core.WEATHER_COLS}
    if not required.issubset(df.columns):
        raise ValueError(f'Archive CSV missing {sorted(required-set(df.columns))}')
    df['issued_at'] = core.parse_timestamps(df.issued_at, 'UTC', 'archive.issued_at')
    df['timestamp'] = core.parse_timestamps(df.timestamp, 'UTC', 'archive.timestamp').dt.floor('h')
    df['location_id'] = df.location_id.astype(str)
    for col in core.WEATHER_COLS[2:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df.duplicated(['issued_at','location_id','timestamp']).any():
        raise ValueError('Archive has duplicate (issued_at, location_id, timestamp) keys')
    return df.sort_values(['issued_at','location_id','timestamp']).reset_index(drop=True)


def get_archive_run(session, site: str, lat: float, lon: float, issued_at: pd.Timestamp,
                    cache_dir: str) -> pd.DataFrame:
    """Select a known model run available *before* issuance, never a stitched series."""
    run = safely_available_run(issued_at)
    if run.date() < date.fromisoformat(DATA_START):
        raise ValueError(f'Run {run} predates documented {MODEL_ID} archive start {DATA_START}')
    params = {'latitude': round(float(lat), 5), 'longitude': round(float(lon), 5),
              'run': run.strftime('%Y-%m-%dT%H:%M'), 'models': MODEL_ID,
              'hourly': HOURS, 'forecast_days': 10, 'timezone': 'UTC',
              'timeformat': 'unixtime'}
    key = os.getenv('OPEN_METEO_API_KEY')
    if key:
        params['apikey'] = key
    url = os.getenv('OPEN_METEO_SINGLE_RUN_URL', ARCHIVE_URL)
    response = core.json_get(session, url, params, Path(cache_dir))
    g = core.parse_api_hourly(response, site)
    # Archive endpoint returns the actual run horizon; do not claim more coverage.
    if not g.timestamp.gt(issued_at).any():
        raise ValueError(f'No target-hour weather available in run {run} for {site}')
    return g.assign(issued_at=issued_at)


def archive_for_origins(traffic: pd.DataFrame, origins: list[date], tz: str,
                        issue_hour: int, cache_dir: str, archive_csv: str | None) -> pd.DataFrame:
    """Offline reproducible archive CSV, or strictly-as-of API retrieval.

    API requests are per site per issuance; can be substantial for many daily
    origins. Cache prevents duplicate requests. Fetch only selected origins.
    """
    if archive_csv:
        archive = load_archive_csv(archive_csv)
        expected = {issue_time(d, tz, issue_hour).tz_convert('UTC') for d in origins}
        actual = set(archive.issued_at.unique())
        missing = expected - actual
        if missing:
            raise ValueError(f'Archive CSV has no forecasts for {len(missing)} requested issue dates; examples {sorted(map(str,missing))[:3]}')
        return archive[archive.issued_at.isin(expected)].copy()
    session = session_with_retries()
    sites = traffic.groupby('location_id', as_index=False)[['latitude','longitude']].median()
    results=[]
    for dt in origins:
        issued = issue_time(dt,tz,issue_hour).tz_convert('UTC')
        for s in sites.itertuples(index=False):
            results.append(get_archive_run(session, s.location_id, s.latitude, s.longitude,
                                           issued, cache_dir))
        print(f'Archived weather ready: issued {dt} ({len(sites)} sites)',flush=True)
    return pd.concat(results, ignore_index=True)


def observed_weather(traffic: pd.DataFrame, weather_csv: str | None,
                     cache_dir: str) -> pd.DataFrame:
    if weather_csv:
        return core.load_weather(weather_csv)
    # Reconstructed historical conditions are used for weather *preceding* issue.
    # They should be replaced with an actual as-of observed archive if needed.
    session = session_with_retries()
    sites = traffic.groupby('location_id',as_index=False)[['latitude','longitude']].median()
    begin=(traffic.timestamp.min()-pd.Timedelta(days=5)).date()
    end=min(pd.Timestamp.now(tz='UTC').date()-timedelta(days=2),traffic.timestamp.max().date())
    if end<begin:
        raise ValueError('Observed-history request end precedes beginning')
    out=[]
    url=os.getenv('OPEN_METEO_HISTORY_URL','https://historical-forecast-api.open-meteo.com/v1/forecast')
    for s in sites.itertuples(index=False):
        params={'latitude':round(float(s.latitude),5),'longitude':round(float(s.longitude),5),
                'start_date':str(begin),'end_date':str(end), 'hourly':HOURS,
                'timezone':'UTC','timeformat':'unixtime'}
        if os.getenv('OPEN_METEO_API_KEY'):
            params['apikey']=os.environ['OPEN_METEO_API_KEY']
        out.append(core.parse_api_hourly(core.json_get(session,url,params,Path(cache_dir)),s.location_id))
    return pd.concat(out,ignore_index=True)


def combine_asof_weather(realized: pd.DataFrame, archive: pd.DataFrame, site: str,
                         issued_at: pd.Timestamp, horizon_end: pd.Timestamp) -> pd.DataFrame:
    """Past observed weather through issue + archived forecast *after* issue.

    No actual weather from a future target is used. Weather rolling features are
    calculated on this single coherent as-of time series.
    """
    start = issued_at - pd.Timedelta(hours=80)
    past = realized[(realized.location_id==site) &
                    (realized.timestamp>=start) & (realized.timestamp<=issued_at)].copy()
    forecast=archive[(archive.location_id==site) & (archive.issued_at==issued_at) &
                     (archive.timestamp>issued_at) & (archive.timestamp<=horizon_end)].copy()
    if forecast.empty:
        raise ValueError(f'Archive missing site {site} at issuance {issued_at}')
    joined=pd.concat([past,forecast[core.WEATHER_COLS]],ignore_index=True)
    joined=joined.drop_duplicates(['location_id','timestamp'],keep='last')
    if joined.empty:
        raise ValueError('Empty as-of weather series')
    return core.weather_features(joined)


def make_rows(traffic: pd.DataFrame, realized: pd.DataFrame, archive: pd.DataFrame,
              origins: list[date], tz: str, hour: int, horizon: int,
              open_hour: int, close_hour: int, closures: set,
              labeled: bool, issued_overrides: dict[date,pd.Timestamp] | None = None) -> pd.DataFrame:
    """For every issuance, build horizon-specific rows using only as-of weather.

    For training, join target washes only where a row exists. No imputed labels.
    Target-hour weather is drawn strictly from archived issuance-specific runs.
    """
    sites=traffic.groupby('location_id',as_index=False)[['latitude','longitude']].median()
    target = traffic[['location_id','timestamp','carwashes']]
    result=[]
    for d in origins:
        issued=(issued_overrides or {}).get(d,issue_time(d,tz,hour).tz_convert('UTC'))
        times=core.future_hours(d+timedelta(days=1),horizon,tz,open_hour,close_hour)
        if len(times)==0:
            continue
        for s in sites.itertuples(index=False):
            grid=pd.DataFrame({'timestamp':times,'location_id':s.location_id,
                               'latitude':s.latitude,'longitude':s.longitude})
            grid=core.add_calendar_features(grid,tz)
            grid=grid[~grid.local_date.map(lambda ld:(s.location_id,ld) in closures)].copy()
            if grid.empty:
                continue
            weather=combine_asof_weather(realized, archive, s.location_id, issued,
                                         grid.timestamp.max()+pd.Timedelta(hours=1))
            grid=grid.merge(weather,on=['location_id','timestamp'],how='left',validate='one_to_one')
            if grid[['temperature_2m','precipitation','cloud_cover']].isna().any(axis=None):
                # Fail instead of silently turning unavailable target forecasts into imputation.
                raise ValueError(f'Missing target weather at site {s.location_id}, origin {d}; verify archived run horizon/coverage')
            grid['issued_at']=issued
            grid['lead_day']=(pd.to_datetime(grid.local_date)-pd.Timestamp(d)).dt.days.astype(int)
            if labeled:
                grid=grid.merge(target,on=['location_id','timestamp'],how='inner',validate='one_to_one')
                if grid.empty:
                    continue
            result.append(grid)
    if not result:
        raise ValueError('No forecasting rows created; verify date ranges and archive coverage')
    return pd.concat(result,ignore_index=True)


def active_traffic(args) -> tuple[pd.DataFrame,set]:
    traffic=core.load_traffic(args.traffic,args.timezone)
    closed=core.load_closures(args.closures)
    loc=traffic.timestamp.dt.tz_convert(args.timezone)
    keep=(loc.dt.hour>=args.open_hour)&(loc.dt.hour<args.close_hour)
    traffic=traffic[keep].copy()
    loc=traffic.timestamp.dt.tz_convert(args.timezone)
    traffic=traffic[~pd.Series(list(zip(traffic.location_id,loc.dt.date)),index=traffic.index).isin(closed)].copy()
    if traffic.empty:
        raise ValueError('No traffic after operating-hour and closure filters')
    return traffic,closed


def dates_for_fit(traffic: pd.DataFrame, args) -> list[date]:
    local=traffic.timestamp.dt.tz_convert(args.timezone)
    last=max(local.dt.date)-timedelta(days=1)  # require at least one known next day
    end=min(last,date.fromisoformat(args.end_date)) if args.end_date else last
    start=date.fromisoformat(args.start_date) if args.start_date else end-timedelta(days=89)
    if start>end:
        raise ValueError('Start date must be no later than end date')
    dates=[]
    d=start
    while d<=end:
        dates.append(d)
        d+=timedelta(days=args.origin_step)
    if not dates:
        raise ValueError('No forecast origins in selected range')
    return dates


def save_meta(meta: dict,folder: Path):
    (folder/'model_info.json').write_text(json.dumps(meta,indent=2,default=str))


def fit(args):
    data,closures=active_traffic(args)
    origins=dates_for_fit(data,args)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    (out/'models').mkdir(exist_ok=True)
    (out/'data_quality.json').write_text(json.dumps(
        core.quality_report(data,closures,args.timezone,args.open_hour,args.close_hour),indent=2))
    # The source CSV can contain years, but API/history need only this selected training window.
    min_needed=issue_time(origins[0],args.timezone,args.issue_hour).tz_convert('UTC')-pd.Timedelta(days=5)
    max_needed=issue_time(origins[-1],args.timezone,args.issue_hour).tz_convert('UTC')+pd.Timedelta(days=args.forecast_days+2)
    selected=data[(data.timestamp>=min_needed)&(data.timestamp<=max_needed)].copy()
    realized=observed_weather(selected,args.weather_csv,args.cache_dir)
    archive=archive_for_origins(selected,origins,args.timezone,args.issue_hour,args.cache_dir,args.archive_csv)
    rows=make_rows(selected,realized,archive,origins,args.timezone,args.issue_hour,
                   args.forecast_days,args.open_hour,args.close_hour,closures,True)
    # Strict cutoff: training targets must all occur before the earliest validation issue.
    if not 1<=args.validation_origins<len(origins):
        raise ValueError('Validation origins must be >= 1 and fewer than the total origins')
    validation_origins=origins[-args.validation_origins:]
    first_validation_issue=issue_time(validation_origins[0],args.timezone,args.issue_hour).tz_convert('UTC')
    train_mask=(rows.issued_at<first_validation_issue)&(rows.timestamp<first_validation_issue)
    validation_mask=(rows.issued_at>=first_validation_issue)
    rows['split']=np.select([train_mask,validation_mask],['train','validation'],default='purged')
    rows.to_csv(out/'asof_training_examples.csv',index=False)
    train=rows[train_mask].copy()
    val=rows[validation_mask].copy()
    if len(train)<100 or len(val)<30:
        raise ValueError(f'Insufficient disjoint training/validation rows: {len(train)} / {len(val)}; shorten validation or enlarge history')
    models=[x.strip() for x in args.models.split(',') if x.strip()]
    if len(set(models))!=len(models) or any(m not in core.ALL_MODELS for m in models):
        raise ValueError('Invalid or duplicate --models setting')
    details=[]
    preds=val[['issued_at','location_id','timestamp','local_date','hour','lead_day','carwashes']].copy()
    preds['split']='validation'
    for name in models:
        print(f'Validation fitting {name} with {len(train):,} rows; testing {len(val):,}...',flush=True)
        model=core.fit_model(name,train)
        yp=core.predict_model(name,model,val)
        details.append({'model':name,**core.model_metrics(val.carwashes.to_numpy(),yp)})
        preds[f'pred_{name}']=np.round(yp,2)
    results=pd.DataFrame(details).sort_values('mae').reset_index(drop=True)
    winner=str(results.iloc[0].model)
    results.to_csv(out/'model_comparison.csv',index=False)
    preds.to_csv(out/'validation_predictions.csv',index=False)
    hm=[]
    for name in models:
        for lead,g in preds.groupby('lead_day'):
            hm.append({'model':name,'lead_day':int(lead),
                       **core.model_metrics(g.carwashes.to_numpy(),g[f'pred_{name}'].to_numpy())})
    pd.DataFrame(hm).to_csv(out/'horizon_metrics.csv',index=False)
    # Final refit can use all labeled rows. Store every candidate for visual comparison.
    for name in models:
        print(f'Final fitting {name} on {len(rows):,} as-of rows...',flush=True)
        joblib.dump(core.fit_model(name,rows),out/'models'/f'{name}.joblib')
    # Track every trained (issued_at, site, target_hour) so future updates cannot
    # accidentally train the same labeled example twice.
    labeled_keys(rows).to_csv(out/'model_training_keys.csv', index=False)
    completed_at = utc_now()
    meta={'version':'0.2.1','best_model':winner,'models':models,
          'last_full_fit_at_utc': completed_at, 'last_incremental_update_at_utc': None,
          'incremental_updates': 0, 'last_model_change_at_utc': completed_at,
          'runtime_versions': runtime_versions(),
          'timezone':args.timezone,'issue_hour':args.issue_hour,
          'open_hour':args.open_hour,'close_hour':args.close_hour,
          'forecast_days':args.forecast_days,
          'origin_dates':[str(origins[0]),str(origins[-1])],
          'origin_step_days':args.origin_step,'validation_origins':[str(x) for x in validation_origins],
          'last_training_target_utc':str(rows.timestamp.max()),
          'last_training_issued_at_utc':str(rows.issued_at.max()),
          'site_count':int(rows.location_id.nunique()),'training_examples':len(train),
          'validation_examples':len(val),'all_labeled_examples':len(rows),
          'weather_mode':'archived_as_of_from_csv' if args.archive_csv else 'single_runs_api',
          'history_mode':'supplied_weather_csv' if args.weather_csv else 'reconstructed_historical',
          'limitations':[
             'Historical pre-issuance weather uses a supplied/reconstructed history, not a frozen real-time observation feed.',
             'Rows can repeat a traffic target at different forecast origins; validation metrics are forecast-level errors.',
             'Validation targets and training targets are separated chronologically, but final refit uses all labeled rows.',
             'No forecast intervals are represented as calibrated probabilities.',
             'Model accuracy on synthetic data is not evidence of real-world accuracy.'
          ]}
    save_meta(meta,out)
    print(f'Trained {len(models)} models; selected {winner}; files in {out.resolve()}')


def get_model_artifacts(folder: Path, names: list[str] | None=None):
    meta=json.loads((folder/'model_info.json').read_text())
    chosen=names or meta['models']
    for m in chosen:
        if m not in meta['models']:
            raise ValueError(f'Model {m} was not included in the saved training run')
    return meta,{m:load_saved_model(folder, m, meta) for m in chosen}


def asof_forward_weather(traffic, origin: date, tz, issue_hour,
                         weather_csv,archive_csv,cache_dir, forecast_days):
    """Offline: use supplied archive at this origin. Online: use live forecast.

    With the live weather API, the weather is available *now*, not necessarily
    at a historical 6pm issue; do not mislabel it as the selected archived run.
    """
    if archive_csv:
        if not weather_csv:
            raise ValueError('--archive-csv on forecast requires --weather-csv for pre-issuance history')
        return core.load_weather(weather_csv),archive_for_origins(
            traffic,[origin],tz,issue_hour,cache_dir,archive_csv),'offline archived forecast',issue_time(origin,tz,issue_hour).tz_convert('UTC')
    if weather_csv:
        # Explicitly not an as-of forecast. Useful to debug offline weather joins only.
        obs=core.load_weather(weather_csv)
        issued=issue_time(origin,tz,issue_hour).tz_convert('UTC')
        future=obs[obs.timestamp>issued].copy().assign(issued_at=issued)
        return obs,future,'offline future-weather oracle (diagnostic only)',issued
    if origin!=pd.Timestamp.now(tz=tz).date():
        raise ValueError('Live weather is only for TODAY as origin; use --archive-csv for past origins')
    # The latest seven days are sufficient for rain lags; avoid fetching years of history.
    recent=traffic[traffic.timestamp>=traffic.timestamp.max()-pd.Timedelta(days=8)].copy()
    live=core.api_weather(recent,forecast_days,cache_dir)
    issued=pd.Timestamp.now(tz='UTC').floor('h')
    # Live predictions use the request-time forecast, rather than pretending it
    # was available at the fixed historic 6 PM training issuance.
    future=live[live.timestamp>issued].copy().assign(issued_at=issued)
    return live,future,'current live weather forecast (as of request)',issued


def forecast(args):
    folder=Path(args.model_dir)
    info,models=get_model_artifacts(folder,
        [args.model] if args.model else None)
    traffic,closures=active_traffic(args)
    tz=info['timezone']
    if args.timezone and args.timezone!=tz:
        raise ValueError(f'Model timezone is {tz}; requested {args.timezone}')
    origin=date.fromisoformat(args.origin_date) if args.origin_date else pd.Timestamp.now(tz=tz).date()
    weather,archive,weather_source,issued_at=asof_forward_weather(
        traffic,origin,tz,int(info['issue_hour']),args.weather_csv,args.archive_csv,
        args.cache_dir,int(info['forecast_days']))
    rows=make_rows(traffic,weather,archive,[origin],tz,int(info['issue_hour']),
                   int(info['forecast_days']),int(info['open_hour']),
                   int(info['close_hour']),closures,False,{origin:issued_at})
    # Forecast rows must not use labels; all as-of inputs should come from weather/cals.
    result=rows[['issued_at','location_id','timestamp','local_date','hour','lead_day',
                 'temperature_2m','precipitation']].copy()
    for name,model in models.items():
        result[f'pred_{name}']=np.round(core.predict_model(name,model,rows),2)
    best=info['best_model'] if info['best_model'] in models else next(iter(models))
    result['expected_washes']=result[f'pred_{best}']
    result['revenue_proxy']=np.round(result.expected_washes*args.revenue_per_wash,2)
    result['local_time']=result.timestamp.dt.tz_convert(tz).astype(str)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    result.to_csv(out/'hourly_forecast.csv',index=False)
    group=result.groupby(['location_id','local_date','lead_day'],as_index=False)
    columns={'expected_washes':('expected_washes','sum'),'revenue_proxy':('revenue_proxy','sum')}
    columns.update({f'pred_{name}':(f'pred_{name}','sum') for name in models})
    daily=group.agg(**columns)
    daily.to_csv(out/'daily_forecast.csv',index=False)
    (out/'forecast_info.json').write_text(json.dumps({
        'model':best,'models':list(models),'weather_source':weather_source,
        'revenue_per_wash_assumption':args.revenue_per_wash,
        'revenue_warning':'Revenue is an illustrative car-count times assumed dollars per wash, not a true company revenue forecast.',
        'asof_limitation':'Historical observed weather before issuance is reconstructed/provided; live inputs reflect request-time updates.',
        'origin_local':str(origin),'actual_weather_issue_utc':str(issued_at),
        'model_last_full_fit_at_utc':info.get('last_full_fit_at_utc'),
        'model_last_incremental_update_at_utc':info.get('last_incremental_update_at_utc'),
        'model_last_training_target_utc':info.get('last_training_target_utc'),
        'hourly_records':int(len(result))},indent=2))
    print(f'Predicted {len(result)} location-hours; model={best}; saved {out.resolve()}')


def update(args):
    """Incrementally update trained MLP only. Other estimators need full refit.

    An MLP's fixed transformer is reused; new site/category values unseen at
    original fit are not safely learnable without a full refit.
    """
    folder=Path(args.model_dir)
    meta=json.loads((folder/'model_info.json').read_text())
    if 'mlp' not in meta['models']:
        raise ValueError('Model folder does not contain mlp; train it or use full fit')
    data,closures=active_traffic(args)
    origins=dates_for_fit(data,args)
    min_needed=issue_time(origins[0],args.timezone,args.issue_hour).tz_convert('UTC')-pd.Timedelta(days=5)
    max_needed=issue_time(origins[-1],args.timezone,args.issue_hour).tz_convert('UTC')+pd.Timedelta(days=args.forecast_days+2)
    selected=data[(data.timestamp>=min_needed)&(data.timestamp<=max_needed)].copy()
    weather=observed_weather(selected,args.weather_csv,args.cache_dir)
    archive=archive_for_origins(selected,origins,meta['timezone'],int(meta['issue_hour']),
                                args.cache_dir,args.archive_csv)
    rows=make_rows(selected,weather,archive,origins,meta['timezone'],int(meta['issue_hour']),
                   int(meta['forecast_days']),int(meta['open_hour']),int(meta['close_hour']),closures,True)
    # Filter at the exact issued-at / site / target-hour level.  Newly observed
    # targets from a previously partially labeled issue are still eligible.
    prior = saved_labeled_keys(folder)
    now = labeled_keys(rows)
    unseen = now.merge(prior, on=KEY_COLS, how='left', indicator=True)
    fresh_keys = unseen.loc[unseen['_merge'].eq('left_only'), KEY_COLS]
    if fresh_keys.empty:
        raise ValueError('No new labeled forecast examples: all available rows have already been learned')
    to_use = fresh_keys.assign(_new=True)
    keyified = rows[KEY_COLS].copy()
    keyified['issued_at'] = pd.to_datetime(keyified.issued_at, utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    keyified['timestamp'] = pd.to_datetime(keyified.timestamp, utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    keyified['location_id'] = keyified.location_id.astype(str)
    rows=rows[keyified.merge(to_use, on=KEY_COLS, how='left')['_new'].fillna(False).to_numpy()].copy()
    model=load_saved_model(folder,'mlp',meta)
    existing_sites=set(map(str,model.named_steps['features'].named_transformers_['cat'].categories_[0]))
    unseen=set(rows.location_id.unique())-existing_sites
    if unseen:
        raise ValueError(f'New sites {unseen} require full refit; one-hot preprocessing is frozen')
    transformed=model.named_steps['features'].transform(core.feature_matrix(rows))
    if model.named_steps['model'].early_stopping:
        raise ValueError('Saved MLP has early_stopping enabled and cannot be updated incrementally')
    model.named_steps['model'].partial_fit(transformed,rows.carwashes.to_numpy(dtype=float))
    # Write the model first and persist the exact processed example keys.
    joblib.dump(model,folder/'models'/'mlp.joblib')
    keys = pd.concat([prior, labeled_keys(rows)], ignore_index=True).drop_duplicates()
    keys.to_csv(folder/'model_training_keys.csv', index=False)
    meta['last_training_issued_at_utc']=str(max(pd.Timestamp(meta['last_training_issued_at_utc']),rows.issued_at.max()))
    meta['last_training_target_utc']=str(max(pd.Timestamp(meta['last_training_target_utc']),rows.timestamp.max()))
    meta['incremental_updates']=int(meta.get('incremental_updates',0))+1
    meta['last_incremental_update_at_utc']=utc_now()
    meta['last_model_change_at_utc']=meta['last_incremental_update_at_utc']
    meta['runtime_versions']=runtime_versions()
    meta['after_update_warning']='Incremental MLP update has NOT been independently revalidated; compare to baseline and retrain other models periodically.'
    save_meta(meta,folder)
    print(f'Updated MLP with {len(rows)} new labeled forecasts. Other models are unchanged; retrain periodically.')


def make_parser():
    p=argparse.ArgumentParser(description='Forecast-origin-aware carwash prediction pipeline v0.2.1')
    sub=p.add_subparsers(dest='command',required=True)
    common=argparse.ArgumentParser(add_help=False)
    common.add_argument('--traffic',required=True)
    common.add_argument('--weather-csv',help='Offline historical observed weather for known dates (optional)')
    common.add_argument('--archive-csv',help='Offline archive of issued weather forecasts (optional); otherwise API')
    common.add_argument('--closures',help='Optional CSV location_id,date of full-day closures')
    common.add_argument('--cache-dir',default='.weather_cache')
    common.add_argument('--timezone',default='America/Chicago')
    common.add_argument('--open-hour',type=int,default=7)
    common.add_argument('--close-hour',type=int,default=19)
    train_parent=argparse.ArgumentParser(add_help=False,parents=[common])
    train_parent.add_argument('--start-date',help='First local forecast ISSUE date, YYYY-MM-DD')
    train_parent.add_argument('--end-date',help='Last local forecast ISSUE date, YYYY-MM-DD')
    train_parent.add_argument('--origin-step',type=int,default=7,help='Days between training origins; choose 1 for daily (more API calls)')
    train_parent.add_argument('--issue-hour',type=int,default=18)
    train_parent.add_argument('--forecast-days',type=int,default=7)
    fit_p=sub.add_parser('fit',parents=[train_parent],help='Strict archived-weather train, validate and refit each model')
    fit_p.add_argument('--validation-origins',type=int,default=2,help='Last N forecast origins used only for validation')
    fit_p.add_argument('--models',default='baseline,poisson,negbin,boosted,mlp')
    fit_p.add_argument('--output',default='asof_results')
    f=sub.add_parser('forecast',parents=[common],help='Forecast from saved models without retraining')
    f.add_argument('--model-dir',default='asof_results')
    f.add_argument('--model',help='Only predict with this saved model')
    f.add_argument('--origin-date',help='Local issuance date for testing, default today')
    f.add_argument('--revenue-per-wash',type=float,default=0.0,help='Assumption; only revenue proxy, not real accounting revenue')
    f.add_argument('--output',default='forward_results')
    upd=sub.add_parser('update',parents=[train_parent],help='Incrementally update MLP with never-before-trained labeled examples')
    upd.add_argument('--model-dir',default='asof_results')
    status=sub.add_parser('status',help='Show when trained, latest included label, and software versions')
    status.add_argument('--model-dir',default='asof_results')
    return p


def main(argv=None):
    args=make_parser().parse_args(argv)
    if args.command=='status':
        meta=json.loads((Path(args.model_dir)/'model_info.json').read_text())
        print(json.dumps({k: meta.get(k) for k in (
            'models','best_model','last_full_fit_at_utc','last_incremental_update_at_utc',
            'incremental_updates','last_model_change_at_utc','last_training_target_utc',
            'last_training_issued_at_utc','runtime_versions')},indent=2))
        return
    if not 0<=args.open_hour<args.close_hour<=24:
        raise ValueError('Operating hours must satisfy 0 <= open < close <= 24')
    if args.command in ('fit','update'):
        if args.command=='update':
            prior=json.loads((Path(args.model_dir)/'model_info.json').read_text())
            for key in ('timezone','open_hour','close_hour','issue_hour','forecast_days'):
                if getattr(args,key)!=prior[key]:
                    raise ValueError(f'Update setting {key} must equal saved model setting {prior[key]}; full refit instead')
        if args.origin_step<1 or not (1<=args.forecast_days<=7):
            raise ValueError('origin-step >=1 and forecast-days 1..7 required')
    if args.command=='fit':
        fit(args)
    elif args.command=='forecast':
        if args.revenue_per_wash<0:
            raise ValueError('Revenue-per-wash assumption must be nonnegative')
        forecast(args)
    elif args.command=='update':
        update(args)

if __name__=='__main__':
    main()
