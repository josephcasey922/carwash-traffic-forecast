"""Pure, testable data logic for a read-only WashCast historical forecast viewer.

Inputs are saved *as-issued* forecast CSVs. No model training or future data access.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = {'issued_at','location_id','timestamp','local_date','hour','lead_day',
            'carwashes','model','prediction'}
FORECAST_KEY = ['issued_at','location_id','timestamp','lead_day','model']
INSTANCE_KEY = ['issued_at','location_id','timestamp','lead_day']
RANGES = ('0–17','18–40','41–60','61+')
WEEKLY_PREFIX = 'weekly_sun__'

ADAPTATION_REQUIRED = {'issued_at','location_id','local_date','hour',
                       'carwashes','candidate','adjusted_prediction'}


def normalize_prediction_schema(raw: pd.DataFrame, source: str = 'source') -> pd.DataFrame:
    """Accept either full historical forecasts or the Day-1 adaptation experiment.

    Adaptation outputs deliberately omit timestamp/lead_day/model and retain BOTH
    the base prediction and adjusted prediction. The latter is what must be
    compared with actuals. Build target UTC timestamps from Chicago local time,
    never by assuming a fixed UTC offset (DST changes during the year).
    """
    if REQUIRED.issubset(raw.columns):
        return raw
    if not ADAPTATION_REQUIRED.issubset(raw.columns):
        missing = sorted(REQUIRED - set(raw.columns))
        raise ValueError(f'{source}: not a prediction file; missing columns: {missing}')
    d = raw.copy()
    d['model'] = 'adaptation_' + d['candidate'].astype('string').str.strip()
    d['prediction'] = d['adjusted_prediction']
    d['lead_day'] = 1
    day = pd.to_datetime(d['local_date'],errors='coerce').dt.normalize()
    hour = pd.to_numeric(d['hour'],errors='coerce')
    target_local_naive = day + pd.to_timedelta(hour,unit='h')
    d['timestamp'] = target_local_naive.dt.tz_localize(
        'America/Chicago',ambiguous='raise',nonexistent='raise').dt.tz_convert('UTC')
    issued_local_day = pd.to_datetime(d['issued_at'],utc=True,errors='coerce').dt.tz_convert(
        'America/Chicago').dt.normalize().dt.tz_localize(None)
    invalid_lead = (day - issued_local_day).ne(pd.Timedelta(days=1))
    if invalid_lead.any():
        raise ValueError(f'{source}: {int(invalid_lead.sum())} adaptation rows are not next-day forecasts')
    return d


def prepare_predictions(raw: pd.DataFrame, source: str = 'source') -> pd.DataFrame:
    raw = normalize_prediction_schema(raw,source)
    missing = sorted(REQUIRED - set(raw.columns))
    if missing:
        raise ValueError(f'{source}: not a prediction file; missing columns: {missing}')
    out = raw.copy()
    out['source_run'] = source
    out['issued_at'] = pd.to_datetime(out.issued_at,utc=True,errors='coerce')
    out['timestamp'] = pd.to_datetime(out.timestamp,utc=True,errors='coerce')
    out['local_date'] = pd.to_datetime(out.local_date,errors='coerce').dt.normalize()
    out['hour'] = pd.to_numeric(out.hour,errors='coerce')
    out['lead_day'] = pd.to_numeric(out.lead_day,errors='coerce')
    out['carwashes'] = pd.to_numeric(out.carwashes,errors='coerce')
    out['prediction'] = pd.to_numeric(out.prediction,errors='coerce')
    out['location_id'] = out.location_id.astype('string').str.strip()
    out['model'] = out.model.astype('string').str.strip()
    invalid = out[['issued_at','timestamp','local_date','hour','lead_day',
                   'carwashes','prediction','location_id','model']].isna().any(axis=1)
    invalid |= out['carwashes'].lt(0) | out['prediction'].lt(0)
    invalid |= out['hour'].lt(0) | out['hour'].gt(23)
    invalid |= out['lead_day'].lt(1) | out['lead_day'].gt(30)
    if invalid.any():
        raise ValueError(f'{source}: {int(invalid.sum())} invalid rows (null/negative/out-of-range value)')
    out['hour'] = out.hour.astype('int16')
    out['lead_day'] = out.lead_day.astype('int16')
    if 'inferred_zero' in out:
        vals = out['inferred_zero'].astype('string').str.lower().str.strip()
        if not vals.isin(['true','false','1','0']).all():
            raise ValueError(f'{source}: invalid/missing inferred_zero flag')
        out['inferred_zero'] = vals.isin(['true','1'])
    else:
        # Unknown provenance; cannot claim all hours were recorded.
        out['inferred_zero'] = pd.Series(pd.NA,index=out.index,dtype='boolean')
    out['abs_error']=(out.prediction-out.carwashes).abs()
    out['signed_error']=out.prediction-out.carwashes
    out['actual_bucket']=bucket(out.carwashes)
    out['predicted_bucket']=bucket(out.prediction)
    out['actual_recommended_employees']=(out['actual_bucket']+2).astype('int8')
    out['predicted_recommended_employees']=(out['predicted_bucket']+2).astype('int8')
    out['employee_difference']=(out['predicted_recommended_employees']-out['actual_recommended_employees']).astype('int8')
    return out


def employee_counts(washes) -> np.ndarray:
    """Recommendation implied by an hourly wash count (not actual scheduled staff).

    Matches bucket_alignment_report.py: nonnegative forecasts are rounded to
    nearest whole wash, with exact half values rounded upward. Thus 17.50 is
    assigned to the 18-40 bucket. Current levels: 2, 3, 4 or 5 employees.
    """
    values=np.asarray(washes,dtype=float)
    rounded=np.floor(np.maximum(values,0)+0.5)
    return (2+np.searchsorted([18,41,61],rounded,side='right')).astype('int8')


def bucket(values) -> pd.Series:
    return pd.Series(employee_counts(values)-2,
                     index=getattr(values,'index',None),dtype='int8')


def combine_predictions(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame,int]:
    if not frames:
        raise ValueError('No prediction CSVs selected')
    df=pd.concat(frames,ignore_index=True)
    # Same actual count must appear under each model for a given as-issued target.
    actual_check=df.groupby(INSTANCE_KEY,observed=True).carwashes.agg(['min','max'])
    if (actual_check['max']-actual_check['min']>1e-6).any():
        raise ValueError('Conflicting actual traffic labels across selected runs. Select one consistent dataset.')
    duplicates=df.duplicated(FORECAST_KEY,keep=False)
    count=int(df.duplicated(FORECAST_KEY).sum())
    if duplicates.any():
        spread=df.loc[duplicates].groupby(FORECAST_KEY,observed=True).prediction.agg(['min','max'])
        if (spread['max']-spread['min']>1e-4).any():
            raise ValueError('Overlapping runs have different forecasts for the same model/issuance/hour. '
                             'Select only one run for each overlapping period.')
    df=df.drop_duplicates(FORECAST_KEY,keep='first').copy()
    return df,count


def quality_cohort(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == 'All (official)':
        return df
    if df.inferred_zero.isna().any():
        raise ValueError('Some selected prediction files lack inferred_zero provenance')
    if name == 'Recorded hours only':
        return df.loc[~df.inferred_zero].copy()
    if name == 'Store-days with no inferred zeros':
        key=['model','issued_at','location_id','local_date','lead_day']
        any_zero=df.groupby(key,observed=True).inferred_zero.transform('any')
        return df.loc[~any_zero].copy()
    if name == 'Exclude store-days with 4+ inferred zeros':
        key=['model','issued_at','location_id','local_date','lead_day']
        n_zero=df.groupby(key,observed=True).inferred_zero.transform('sum')
        return df.loc[n_zero<4].copy()
    raise ValueError(f'Unknown cohort: {name}')


def metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(n=0,mae=np.nan,median_ae=np.nan,wape=np.nan,bias=np.nan,
                    p90_ae=np.nan,within_5=np.nan,within_10=np.nan,
                    bucket_agreement=np.nan,bucket_under=np.nan,bucket_over=np.nan,
                    forecast_labor_alignment_percent=np.nan,
                    recommended_employee_hours=0,absolute_employee_hour_difference=0,
                    under_recommended_employee_hours=0,over_recommended_employee_hours=0,
                    mean_actual=np.nan,inferred_fraction=np.nan)
    abs_err=np.abs(df.prediction.to_numpy(float)-df.carwashes.to_numpy(float))
    signed=df.prediction.to_numpy(float)-df.carwashes.to_numpy(float)
    y=df.carwashes.to_numpy(float)
    ab=df.actual_bucket.to_numpy();pb=df.predicted_bucket.to_numpy()
    employee_difference=pb.astype('int64')-ab.astype('int64')
    recommended_employee_hours=int((ab.astype('int64')+2).sum())
    absolute_employee_hour_difference=int(np.abs(employee_difference).sum())
    return dict(n=int(len(df)),mae=float(abs_err.mean()),median_ae=float(np.median(abs_err)),
                wape=float(100*abs_err.sum()/y.sum()) if y.sum()>0 else np.nan,
                bias=float(signed.mean()),p90_ae=float(np.quantile(abs_err,.9)),
                within_5=float(100*(abs_err<=5).mean()),
                within_10=float(100*(abs_err<=10).mean()),
                bucket_agreement=float(100*(pb==ab).mean()),
                bucket_under=float(100*(pb<ab).mean()),
                bucket_over=float(100*(pb>ab).mean()),
                forecast_labor_alignment_percent=float(100*(1-absolute_employee_hour_difference/recommended_employee_hours)),
                recommended_employee_hours=recommended_employee_hours,
                absolute_employee_hour_difference=absolute_employee_hour_difference,
                under_recommended_employee_hours=int(np.maximum(-employee_difference,0).sum()),
                over_recommended_employee_hours=int(np.maximum(employee_difference,0).sum()),
                mean_actual=float(y.mean()),
                inferred_fraction=float(df.inferred_zero.mean()*100) if not df.inferred_zero.isna().any() else np.nan)


def daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Daily total metrics only when ALL scheduled hours are present; no partial-day scoring."""
    if df.empty:return pd.DataFrame(columns=['model','location_id','local_date','lead_day','actual','predicted','hours','expected_hours','complete','signed_error','abs_error'])
    keys=['model','location_id','local_date','lead_day']
    a=df.groupby(keys,observed=True).agg(actual=('carwashes','sum'),predicted=('prediction','sum'),
                hours=('hour','nunique'),min_hour=('hour','min'),max_hour=('hour','max'),
                inferred_hours=('inferred_zero', lambda v: int(v.fillna(False).sum()))).reset_index()
    sunday=a.local_date.dt.dayofweek.eq(6)
    a['expected_hours']=np.where(sunday,10,11)
    a['expected_start']=np.where(sunday,9,8)
    a['complete']=(a.hours==a.expected_hours)&(a.min_hour==a.expected_start)&a.max_hour.eq(18)
    a['signed_error']=a.predicted-a.actual
    a['abs_error']=a.signed_error.abs()
    return a


def daily_metrics(daily: pd.DataFrame) -> dict:
    c=daily.loc[daily.complete]
    if c.empty:return dict(days=0,mae=np.nan,median_ae=np.nan,wape=np.nan,bias=np.nan,within_20pct=np.nan)
    pct=c.abs_error/c.actual.replace(0,np.nan)
    return dict(days=int(len(c)),mae=float(c.abs_error.mean()),median_ae=float(c.abs_error.median()),
                wape=float(100*c.abs_error.sum()/c.actual.sum()) if c.actual.sum()>0 else np.nan,
                bias=float(c.signed_error.mean()),within_20pct=float(100*pct.le(.2).mean()))


def paired_predictions(df:pd.DataFrame,model_a:str,model_b:str)->pd.DataFrame:
    keys=INSTANCE_KEY
    a=df.loc[df.model.eq(model_a),keys+['carwashes','prediction']].rename(columns={'prediction':'prediction_a'})
    b=df.loc[df.model.eq(model_b),keys+['carwashes','prediction']].rename(columns={'prediction':'prediction_b'})
    merged=a.merge(b,on=keys,how='inner',suffixes=('_a','_b'),validate='one_to_one')
    if not merged.empty and (merged.carwashes_a-merged.carwashes_b).abs().gt(1e-6).any():
        raise ValueError('Comparison models do not have the same actual labels')
    return merged




def matched_model_alignment(frame: pd.DataFrame, models: list[str], *,
                            match_target_hours: bool = False) -> pd.DataFrame:
    """Company-formula alignment for each model on EXACTLY the same hours.

    For ordinary model comparisons, match the issuance + store + target hour +
    lead day; for Sunday-fixed versus daily-refreshed forecasts, pair target
    store-hours (the issuances are deliberately different). Weekly and daily
    percentages MUST each be computed from the SUM of absolute employee-hour
    differences / SUM of actual recommended employee-hours.
    """
    columns=['model','hours','forecast_labor_alignment_percent','bucket_agreement',
             'recommended_employee_hours','absolute_employee_hour_difference',
             'under_recommended_employee_hours','over_recommended_employee_hours',
             'mae','wape','bias']
    models=list(dict.fromkeys(m for m in models if m))
    if not models or frame.empty:
        return pd.DataFrame(columns=columns)
    keys=['location_id','timestamp'] if match_target_hours else INSTANCE_KEY
    available=[]
    for model in models:
        subset=frame.loc[frame.model.eq(model)]
        if subset.empty:
            return pd.DataFrame(columns=columns)
        if subset.duplicated(keys).any():
            raise ValueError(f'{model}: duplicate forecast keys prevent a fair aligned comparison')
        available.append(subset)
    # Inner intersection of hours present for every model.
    matched=available[0][keys].copy()
    for subset in available[1:]:
        matched=matched.merge(subset[keys],on=keys,how='inner',validate='one_to_one')
    if matched.empty:
        return pd.DataFrame(columns=columns)
    aligned=[]
    true_values=None
    for model,subset in zip(models,available):
        rows=matched.merge(subset,on=keys,how='left',validate='one_to_one')
        # Sort by common keys before comparing actual counts across forecast vintages.
        rows=rows.sort_values(keys,kind='stable').reset_index(drop=True)
        actual=rows.carwashes.to_numpy(float)
        if true_values is not None and not np.allclose(actual,true_values,rtol=0,atol=1e-6):
            raise ValueError('Models disagree on actual washes for matched hours; refresh from the same source')
        true_values=actual
        score=metrics(rows)
        aligned.append({'model':model,'hours':score['n'],
            **{k:score[k] for k in columns if k not in ('model','hours')}})
    return pd.DataFrame(aligned,columns=columns)


def matched_alignment_by_period(frame: pd.DataFrame, models: list[str], *,
                                period: str = 'day', match_target_hours: bool = False) -> pd.DataFrame:
    """Daily, weekly, or monthly model-labelled alignment on matched hours.

    The weekly value is calculated using all the week's raw hourly employee
    differences and the week's total actual recommendations; NEVER averaging
    seven daily scores. A partial week is labelled by its matched hour count.
    """
    if period not in ('day','week','month'):
        raise ValueError('period must be day, week, or month')
    result=[]
    if frame.empty:return pd.DataFrame(columns=['period','model','hours','forecast_labor_alignment_percent'])
    d=frame.copy()
    if period=='day':
        d['__period__']=d.local_date.dt.strftime('%Y-%m-%d')
    elif period=='week':
        d['__period__']=week_start(d.local_date).dt.strftime('%Y-%m-%d')
    else:
        d['__period__']=d.local_date.dt.strftime('%Y-%m')
    for label,group in d.groupby('__period__',sort=True,observed=True):
        compare=matched_model_alignment(group,models,match_target_hours=match_target_hours)
        for row in compare.to_dict('records'):
            result.append({'period':label,**row})
    return pd.DataFrame(result)


def week_start(dates:pd.Series)->pd.Series:
    return dates-pd.to_timedelta(dates.dt.dayofweek,unit='D')


def month_summary(df:pd.DataFrame)->pd.DataFrame:
    d=df.copy()
    d['month']=d.local_date.dt.to_period('M').astype(str)
    return pd.DataFrame([{'model':model,'month':month,**metrics(g)}
                         for (model,month),g in d.groupby(['model','month'],observed=True)])


def store_scorecard(df:pd.DataFrame)->pd.DataFrame:
    return pd.DataFrame([{'location_id':site,'model':model,**metrics(g)}
                         for (site,model),g in df.groupby(['location_id','model'],observed=True)])


def discover_prediction_csvs(root:str|Path)->list[Path]:
    path=Path(root).expanduser()
    if not path.exists():return []
    if path.is_file(): return [path] if path.name.endswith('.csv') else []
    found=[]
    for pattern in ('demand_ablation_*/all_predictions.csv',
                    'clean_label_*/all_predictions.csv',
                    'clean_label_backtest_*/all_predictions.csv',
                    'adaptation_*/all_candidate_day1_predictions.csv',
                    'validation_*/all_predictions.csv',
                    'sunday_weekly_*/weekly_predictions.csv'):
        found+=list(path.glob(pattern))
    return sorted(set(p for p in found if p.is_file()))


def prepare_upcoming_weekly(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate future, unscored Sunday-issued forecasts.

    No invented zero actuals; never mix with scored historical predictions.
    """
    keys=['issued_at','location_id','timestamp','local_date','hour','lead_day','prediction','model']
    missing=set(keys)-set(raw.columns)
    if missing:
        raise ValueError(f'Upcoming weekly forecast missing: {sorted(missing)}')
    if 'carwashes' in raw and raw.carwashes.notna().any():
        raise ValueError('Upcoming forecast must not include actual target labels')
    d=raw[keys].copy()
    d['issued_at']=pd.to_datetime(d.issued_at,utc=True,errors='raise')
    d['timestamp']=pd.to_datetime(d.timestamp,utc=True,errors='raise')
    d['local_date']=pd.to_datetime(d.local_date,errors='raise').dt.normalize()
    d['hour']=pd.to_numeric(d.hour,errors='raise')
    d['lead_day']=pd.to_numeric(d.lead_day,errors='raise')
    d['prediction']=pd.to_numeric(d.prediction,errors='raise')
    d['location_id']=d.location_id.astype(str)
    d['model']=d.model.astype(str)
    issue_local=d.issued_at.dt.tz_convert('America/Chicago')
    day_gap=(d.local_date-issue_local.dt.tz_localize(None).dt.normalize()).dt.days
    invalid=(d[keys].isna().any(axis=1) | d.prediction.lt(0) |
             ~np.isfinite(d.prediction) | ~d.lead_day.between(1,7) |
             (day_gap != d.lead_day) | issue_local.dt.dayofweek.ne(6) |
             issue_local.dt.hour.ne(18) | issue_local.dt.minute.ne(0) |
             ~d.model.str.startswith(WEEKLY_PREFIX) | d.timestamp.le(d.issued_at))
    if invalid.any():
        raise ValueError(f'Invalid upcoming Sunday forecast rows: {int(invalid.sum())}')
    if d.duplicated(['model','issued_at','location_id','timestamp']).any():
        raise ValueError('Duplicate upcoming Sunday forecast hours')
    d['hour']=d.hour.astype(int)
    d['lead_day']=d.lead_day.astype(int)
    return d


def matched_daily_weekly(daily: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    """Pair the DIFFERENT vintage forecasts only by target store-hour.

    Daily is lead-day 1, issued one day before target. Weekly was issued on
    the preceding Sunday. Never pretend these share the same issuance time.
    """
    keys=['location_id','timestamp']
    if daily.duplicated(keys).any() or weekly.duplicated(keys).any():
        raise ValueError('Expected one daily and one Sunday-weekly prediction per store-hour')
    a=daily[keys+['carwashes','prediction']].rename(columns={'prediction':'daily_prediction'})
    b=weekly[keys+['carwashes','prediction','lead_day']].rename(columns={'prediction':'weekly_prediction'})
    paired=a.merge(b,on=keys,how='inner',validate='one_to_one',suffixes=('_daily','_weekly'))
    if not paired.empty and (paired.carwashes_daily-paired.carwashes_weekly).abs().gt(1e-6).any():
        raise ValueError('Daily and weekly actual wash counts disagree; rebuild from the same traffic export')
    return paired


def sunday_weekly_rows(weekly: pd.DataFrame, *, model: str, location: str,
                       monday: pd.Timestamp) -> pd.DataFrame:
    """Choose exactly the Sunday 6 PM forecast for the given Mon–Sun week."""
    if weekly.empty:
        return weekly.copy()
    monday=pd.Timestamp(monday).normalize()
    if monday.dayofweek != 0:
        raise ValueError('Expected a Monday week start')
    sunday=(monday-pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    issue_days=weekly.issued_at.dt.tz_convert('America/Chicago').dt.strftime('%Y-%m-%d')
    mask=(weekly.model.eq(model)&weekly.location_id.eq(location)&issue_days.eq(sunday)
          &weekly.local_date.between(monday,monday+pd.Timedelta(days=6)))
    return weekly.loc[mask].copy()


def load_location_names(root:str|Path)->dict[str,str]:
    for path in (Path(root)/'location_id_mapping.csv',Path(root)/'LocationIDToLocation.xlsx'):
        if not path.is_file():continue
        try:
            if path.suffix=='.csv':d=pd.read_csv(path)
            else:d=pd.read_excel(path)
            # The user supplied both a two-column CSV and an original workbook.
            if {'location_id','location_name'}.issubset(d.columns):
                names=dict(zip(d.location_id.astype(str),d.location_name.astype(str)))
                return {k:v for k,v in names.items() if k and v and v!='nan'}
        except Exception:continue
    return {}
