"""WashCast: strictly as-of next-day traffic adaptation experiment.

This is a development experiment around a FROZEN base model, not a staff model.
Every offset for a target date D uses only forecast residuals with target dates
<= D-2, mimicking a forecast issued at ~6 PM on D-1. No weather APIs, refitting,
or target-day observations are used to compute adjustments.

Important: calibrator variants are hypotheses, not proven improvements. Select on
May-July development forecasts; previously reviewed August-September is no
longer a fresh holdout. For deployment, establish data-availability latency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
CANDIDATES = ('base', 'store_fast_14d', 'store_slow_28d',
              'volume_35d', 'fast_plus_volume', 'slow_plus_volume')
THRESHOLDS = np.array([18., 41., 61.], dtype=float)
VOLUME_BANDS = ('<35', '35-50', '>=50')


def buckets(x: np.ndarray) -> np.ndarray:
    """0-17, 18-40, 41-60, 61+ after rounding nonnegative count half up."""
    arr = np.maximum(np.asarray(x,dtype=float), 0)
    counts = np.floor(arr + .5)
    return np.searchsorted(THRESHOLDS, counts, side='right').astype(int)


def volume_band(p: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.array([35.,50.]),np.asarray(p,dtype=float),side='right')


def read_predictions(paths: list[Path], model: str) -> pd.DataFrame:
    parts = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path,dtype={'location_id':str})
        required = {'issued_at','location_id','local_date','hour','lead_day',
                    'carwashes','prediction','model'}
        missing = required - set(frame)
        if missing:
            raise ValueError(f'{path} missing columns: {sorted(missing)}')
        frame = frame.loc[frame.model.eq(model) & (pd.to_numeric(frame.lead_day)==1)].copy()
        if frame.empty:
            raise ValueError(f'{path} has no lead_day=1 predictions for model={model}')
        keep = ['issued_at','location_id','local_date','hour','carwashes','prediction']
        if 'inferred_zero' in frame:
            keep.append('inferred_zero')
        parts.append(frame[keep])
    df = pd.concat(parts,ignore_index=True)
    df['issued_at'] = pd.to_datetime(df.issued_at,utc=True,errors='raise')
    df['local_date'] = pd.to_datetime(df.local_date,errors='raise').dt.normalize()
    df['hour'] = pd.to_numeric(df.hour,errors='raise').astype(int)
    df['carwashes'] = pd.to_numeric(df.carwashes,errors='raise').astype(float)
    df['prediction'] = pd.to_numeric(df.prediction,errors='raise').astype(float)
    if not np.isfinite(df[['carwashes','prediction']].to_numpy()).all():
        raise ValueError('Nonfinite targets or predictions')
    if (df[['carwashes','prediction']] < 0).any().any():
        raise ValueError('Negative wash counts/predictions')
    if 'inferred_zero' not in df:
        df['inferred_zero'] = False
    else:
        df['inferred_zero'] = df.inferred_zero.astype(str).str.lower().eq('true')
    if df.duplicated(['location_id','local_date','hour']).any():
        raise ValueError('Duplicate Day-1 store/date/hour, likely overlapping input files')
    if df.duplicated(['issued_at','location_id','local_date','hour']).any():
        raise ValueError('Repeated forecast keys')
    # Enforce a real next-day issue: a 6 PM Chicago issue on D-1 predicts D.
    local_issued = df.issued_at.dt.tz_convert(ZoneInfo('America/Chicago')).dt.tz_localize(None).dt.normalize()
    if not (df.local_date-local_issued).eq(pd.Timedelta(days=1)).all():
        raise ValueError('Expected all lead_day=1 rows to target the NEXT Chicago calendar day')
    df = df.sort_values(['local_date','location_id','hour']).reset_index(drop=True)
    df['error'] = df.prediction-df.carwashes
    df['absolute_error'] = df.error.abs()
    df['volume_band'] = volume_band(df.prediction.to_numpy())
    return df


def daily_history(d: pd.DataFrame) -> pd.DataFrame:
    """Construct historical completed store-day residuals and quality flags."""
    grouped = (d.groupby(['location_id','local_date'],as_index=False)
               .agg(hours=('hour','nunique'), actual_mean=('carwashes','mean'),
                    observed_total=('carwashes','sum'), prediction_total=('prediction','sum'),
                    mean_error=('error','mean'), inferred=('inferred_zero','sum')))
    grouped['expected_hours'] = np.where(grouped.local_date.dt.dayofweek.eq(6),10,11)
    grouped['eligible'] = (grouped.hours.eq(grouped.expected_hours) & grouped.inferred.lt(3))
    # Network-shock determination for day T uses ONLY T's realized outcomes.
    # It is only consulted for future target days >= T+2.
    perdate = grouped[grouped.eligible].groupby('local_date').agg(
        n=('location_id','nunique'), signs_pos=('mean_error',lambda x:int((x>0).sum())),
        signs_neg=('mean_error',lambda x:int((x<0).sum())),
        actual=('observed_total','sum'),pred=('prediction_total','sum'))
    same = perdate[['signs_pos','signs_neg']].max(axis=1) / perdate.n
    bias = (perdate.pred-perdate.actual) / perdate.actual.replace(0,np.nan)
    shocks = perdate.index[(perdate.n>=12) & same.ge(.93) & bias.abs().ge(.25)]
    grouped['network_shock'] = grouped.local_date.isin(shocks)
    # A questionable day can be scored normally but not used to adjust later forecasts.
    grouped['eligible'] &= ~grouped.network_shock
    return grouped


def store_offset(history: pd.DataFrame, current: pd.Timestamp, store: str,
                 *, span: int, min_days: int, sign_requirement: float,
                 min_washes: float, magnitude_fraction: float,
                 strength: float, cap: float) -> tuple[float,dict]:
    cutoff=current-pd.Timedelta(days=2)
    past=history[(history.location_id==store) & history.eligible
                 & history.local_date.le(cutoff)
                 & history.local_date.ge(current-pd.Timedelta(days=span+1))]
    values=past.mean_error.to_numpy(dtype=float)
    n=len(values)
    median=float(np.median(values)) if n else np.nan
    reference=float(past.actual_mean.mean()) if n else np.nan
    share=float(np.mean(np.sign(values)==np.sign(median))) if n and median!=0 else 0.0
    threshold=max(min_washes,magnitude_fraction*reference) if n else np.nan
    eligible=bool(n>=min_days and share>=sign_requirement and np.abs(median)>=threshold)
    # correction has opposite sign of historical pred - actual error.
    offset=float(np.clip(-strength*median,-cap,cap)) if eligible else 0.0
    return offset,{'history_days':n,'history_median_bias':median,
                   'history_sign_agreement':share,'history_mean_actual':reference,
                   'gate_magnitude':threshold,'triggered':eligible}


def volume_offsets(d: pd.DataFrame, daily: pd.DataFrame, current: pd.Timestamp) -> tuple[np.ndarray,dict]:
    """Pooled count calibration, conditional on *forecasted*, not realized, volume.

    Clip individual historical residuals; use no future target labels. Never
    calibrate the under-35 bin because it was nearly unbiased on initial audit.
    """
    cutoff=current-pd.Timedelta(days=2)
    history=d[(d.local_date<=cutoff) & (d.local_date>=current-pd.Timedelta(days=36))]
    clean=daily[daily.eligible][['location_id','local_date']]
    history=history.merge(clean,on=['location_id','local_date'],how='inner',validate='many_to_one')
    result=np.zeros(3,dtype=float)
    stats={}
    for band in (1,2):
        b=history[history.volume_band.eq(band)]
        n=len(b);days=b.local_date.nunique()
        bias=float(np.clip(b.error.to_numpy(),-25,25).mean()) if n else np.nan
        # Fixed, intentionally conservative. Availability varies by predicted band.
        if n>=120 and days>=10:
            result[band]=float(np.clip(-.6*(n/(n+250))*bias,-3.,3.))
        stats[f'volume_bin_{band}_hours']=n
        stats[f'volume_bin_{band}_days']=int(days)
        stats[f'volume_bin_{band}_historical_bias']=bias
        stats[f'volume_bin_{band}_adjustment']=result[band]
    return result,stats


def apply_candidates(d: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    """All candidates use the same as-of history of BASE-model residuals."""
    day=daily_history(d)
    out=[];history_rows=[]
    for when, rows in d.groupby('local_date',sort=True):
        sites=sorted(rows.location_id.unique())
        v,vs=volume_offsets(d,day,when)
        for site in sites:
            r=rows[rows.location_id==site]
            fast,fs=store_offset(day,when,site,span=14,min_days=7,
                                 sign_requirement=.65,min_washes=.75,
                                 magnitude_fraction=.05,strength=.60,cap=4.)
            slow,ss=store_offset(day,when,site,span=28,min_days=10,
                                 sign_requirement=.70,min_washes=1.,
                                 magnitude_fraction=.08,strength=.50,cap=4.)
            history_rows.append(dict(local_date=when.date().isoformat(),location_id=site,
                fast_offset=fast,slow_offset=slow,
                fast_days=fs['history_days'],slow_days=ss['history_days'],
                fast_median_prior_bias=fs['history_median_bias'],
                slow_median_prior_bias=ss['history_median_bias'],
                fast_gate=fs['triggered'],slow_gate=ss['triggered'],
                **vs))
            base=r.prediction.to_numpy(dtype=float)
            vol=v[r.volume_band.to_numpy(dtype=int)]
            offsets={'base':np.zeros(len(r)),
                     'store_fast_14d':np.full(len(r),fast),
                     'store_slow_28d':np.full(len(r),slow),
                     'volume_35d':vol,
                     'fast_plus_volume':np.clip(fast+vol,-6.,6.),
                     'slow_plus_volume':np.clip(slow+vol,-6.,6.)}
            basic=r[['issued_at','local_date','location_id','hour','carwashes',
                     'prediction','inferred_zero','volume_band']].copy()
            for label,adjustment in offsets.items():
                part=basic.copy()
                part['candidate']=label
                part['offset']=adjustment
                part['adjusted_prediction']=np.maximum(0,base+adjustment)
                part['absolute_error']=np.abs(part.adjusted_prediction-part.carwashes)
                part['signed_error']=part.adjusted_prediction-part.carwashes
                part['actual_bucket']=buckets(part.carwashes.to_numpy())
                part['predicted_bucket']=buckets(part.adjusted_prediction.to_numpy())
                out.append(part)
    predictions=pd.concat(out,ignore_index=True)
    history=pd.DataFrame(history_rows)
    return predictions,history


def summarize(data: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    rows=[]
    for key,g in data.groupby(group,dropna=False,observed=True,sort=True):
        key=key if isinstance(key,tuple) else (key,)
        y=g.carwashes.to_numpy(dtype=float)
        err=g.signed_error.to_numpy(dtype=float)
        ae=np.abs(err)
        pred_b=g.predicted_bucket.to_numpy(dtype=int)
        actual_b=g.actual_bucket.to_numpy(dtype=int)
        rows.append({**dict(zip(group,key)),
            'hours':int(len(g)),'mae':float(ae.mean()),
            'median_absolute_error':float(np.median(ae)),
            'wape_percent':float(100*ae.sum()/y.sum()) if y.sum()>0 else np.nan,
            'signed_bias':float(err.mean()),
            'bucket_agreement_percent':float(100*np.mean(pred_b==actual_b)),
            'bucket_under_percent':float(100*np.mean(pred_b<actual_b)),
            'bucket_over_percent':float(100*np.mean(pred_b>actual_b)),
            'mean_predicted_washes':float(g.adjusted_prediction.mean()),
            'mean_actual_washes':float(y.mean()),
            'actual_61plus_hours':int((y>=61).sum()),
            'forecast_61plus_hours':int((pred_b==3).sum()),
        })
    return pd.DataFrame(rows)


def make_reports(out: pd.DataFrame, hist: pd.DataFrame, score_from: str|None,
                 score_to: str|None) -> dict[str,pd.DataFrame]:
    data=out
    if score_from:
        data=data[data.local_date>=pd.Timestamp(score_from)]
    if score_to:
        data=data[data.local_date<=pd.Timestamp(score_to)]
    if data.empty:
        raise ValueError('No Day-1 forecasts in requested score dates')
    data=data.copy()
    data['forecast_volume_band'] = np.select(
        [data.prediction<35,data.prediction<50],['<35','35-50'],default='>=50')
    data['week_start']=(data.local_date-pd.to_timedelta(data.local_date.dt.dayofweek,unit='D')).dt.date
    all_metrics=summarize(data,['candidate']).sort_values('mae')
    stores=summarize(data,['candidate','location_id'])
    weeks=summarize(data,['candidate','week_start'])
    volumes=summarize(data,['candidate','forecast_volume_band'])
    extreme=summarize(data[data.carwashes>=61],['candidate'])
    if len(extreme)==0:
        extreme=pd.DataFrame(columns=all_metrics.columns)
    clean=summarize(data[~data.inferred_zero],['candidate'])
    # MAE gain vs base over same store-hours, not a test of significance.
    base=stores[stores.candidate.eq('base')][['location_id','mae','bucket_under_percent']]
    stores=stores.merge(base.rename(columns={'mae':'base_mae','bucket_under_percent':'base_bucket_under_percent'}),
                        on='location_id',how='left',validate='many_to_one')
    stores['mae_improvement']=stores.base_mae-stores.mae
    stores['under_bucket_change_pp']=stores.bucket_under_percent-stores.base_bucket_under_percent
    return {'overall_day1.csv':all_metrics,'by_store_day1.csv':stores,
            'by_week_day1.csv':weeks,'by_predicted_volume_day1.csv':volumes,
            'realized_61plus_day1.csv':extreme,
            'excluding_inferred_zero_hour_day1.csv':clean,
            'asof_calibration_history.csv':hist}


def run(*,paths:list[Path],model:str,output:Path,score_from:str|None=None,
        score_to:str|None=None) -> dict[str,pd.DataFrame]:
    d=read_predictions(paths,model)
    corrected,history=apply_candidates(d)
    output.mkdir(parents=True,exist_ok=True)
    corrected.to_csv(output/'all_candidate_day1_predictions.csv',index=False)
    reports=make_reports(corrected,history,score_from,score_to)
    for name,report in reports.items():
        report.to_csv(output/name,index=False)
    manifest={'schema_version':SCHEMA_VERSION,'base_model':model,
              'source_files':[str(path) for path in paths],
              'first_target_date':str(d.local_date.min().date()),
              'last_target_date':str(d.local_date.max().date()),
              'score_from':score_from,'score_to':score_to,
              'candidates':list(CANDIDATES),
              'causality':'Target D uses only validated Day-1 historical target dates <= D-2.',
              'selection_warning':'May-July is development; August-September has been repeatedly examined and must not be used to tune coefficients.',
              'note':'Carwash count is the prediction target; staffing range agreement is an assessment only.'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(reports['overall_day1.csv'][['candidate','hours','mae','wape_percent',
                   'signed_bias','bucket_agreement_percent','bucket_under_percent']].to_string(index=False))
    print('Results:',output.resolve())
    return reports


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',nargs='+',type=Path,required=True,
                   help='One or more nonoverlapping all_predictions.csv files (May-July + August-September, for example)')
    p.add_argument('--model',default='lgbm_stable_recent')
    p.add_argument('--score-from',help='Only score target days on/after date, while using earlier data as training history')
    p.add_argument('--score-to',help='Only score target days on/before date')
    p.add_argument('--output',type=Path,default=Path('adaptive_calibration_dev'))
    a=p.parse_args()
    run(paths=a.predictions,model=a.model,output=a.output,score_from=a.score_from,score_to=a.score_to)


if __name__=='__main__':
    cli()
