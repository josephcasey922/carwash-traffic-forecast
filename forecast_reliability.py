"""Paired WashCast reliability scorecard for multi-location, weather-aware forecasts.

Use with predictions from demand_model_benchmark.py. Not a claim of statistical
independence: store-hours, overlapping 2-day forecasts, and nearby-store storms
are correlated. The 'rainy_store_day' report measures forecast-rain episodes,
NOT independent verified storms or actual observed precipitation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ['issued_at','location_id','timestamp']
REQUIRED = KEYS + ['model','carwashes','prediction','weather_regime','local_date','lead_day']


def validate_predictions(source: pd.DataFrame, *, reference: str = 'weather_median') -> pd.DataFrame:
    if not set(REQUIRED).issubset(source):
        raise ValueError(f'Missing prediction columns: {sorted(set(REQUIRED)-set(source))}')
    df=source.copy()
    df['issued_at']=pd.to_datetime(df.issued_at,utc=True,errors='raise')
    df['timestamp']=pd.to_datetime(df.timestamp,utc=True,errors='raise')
    df['location_id']=df.location_id.astype(str)
    df['model']=df.model.astype(str)
    for c in ('carwashes','prediction'):
        df[c]=pd.to_numeric(df[c],errors='raise')
    if df[['carwashes','prediction']].isna().any().any() or not np.isfinite(df[['carwashes','prediction']]).all().all():
        raise ValueError('Nonfinite actual or prediction')
    if (df[['carwashes','prediction']] < 0).any().any():
        raise ValueError('Negative count or predicted count')
    if df.duplicated(['model']+KEYS).any():
        raise ValueError('Duplicate (model,issued_at,location_id,timestamp)')
    if reference not in df.model.unique():
        raise ValueError(f'Reference model {reference!r} not in predictions')
    ref=df[df.model.eq(reference)][KEYS+['carwashes']]
    for name, part in df.groupby('model',sort=False):
        merged=ref.merge(part[KEYS+['carwashes']],on=KEYS,how='outer',validate='one_to_one',indicator=True,suffixes=('_ref','_new'))
        if not merged._merge.eq('both').all() or not np.allclose(merged.carwashes_ref,merged.carwashes_new):
            raise ValueError(f'Model {name} was not evaluated on identical forecast targets')
    df['absolute_error']=np.abs(df.prediction-df.carwashes)
    df['signed_error']=df.prediction-df.carwashes
    df['issue_date']=df.issued_at.dt.strftime('%Y-%m-%d')
    if 'inferred_zero' in df:
        df['inferred_zero']=df.inferred_zero.astype('string').str.lower().eq('true')
    if 'rd_store_age_days' in df:
        age=pd.to_numeric(df.rd_store_age_days, errors='coerce')
        df['store_age_group']=pd.cut(age,bins=[-1,7,28,90,365,1e6],
                labels=['0-7 days','8-28 days','29-90 days','91-365 days','>365 days'])
        df['store_age_group']=df['store_age_group'].astype(object).fillna('unknown')
    return df


def summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows=[]
    groups=df.groupby(by,dropna=False,sort=True,observed=True) if by else [((),df)]
    for key,part in groups:
        key=key if isinstance(key,tuple) else (key,)
        n=len(part)
        if n==0: continue
        abs_err=part.absolute_error.to_numpy(dtype=float)
        signed=part.signed_error.to_numpy(dtype=float)
        y=part.carwashes.to_numpy(dtype=float)
        rec=dict(zip(by,key))
        rec.update(observations=n,mae=float(abs_err.mean()),
                   rmse=float(np.sqrt(np.mean(signed**2))),
                   p90_absolute_error=float(np.quantile(abs_err,.9)),
                   wape_percent=float(abs_err.sum()/y.sum()*100) if y.sum()>0 else np.nan,
                   signed_bias=float(signed.mean()),
                   actual_average=float(y.mean()),predicted_average=float(part.prediction.mean()))
        rows.append(rec)
    return pd.DataFrame(rows)


def paired_skill(df: pd.DataFrame, groups: list[str], reference: str='weather_median') -> pd.DataFrame:
    ref=(df[df.model.eq(reference)][KEYS+['carwashes','absolute_error']]
         .rename(columns={'carwashes':'ref_actual','absolute_error':'ref_error'}))
    others=df[df.model.ne(reference)]
    joined=others.merge(ref,on=KEYS,how='left',validate='many_to_one')
    if joined.ref_error.isna().any() or not np.allclose(joined.carwashes,joined.ref_actual):
        raise ValueError('Reference model targets do not align')
    by=['model']+groups
    rows=[]
    for label,g in joined.groupby(by,dropna=False,observed=True,sort=True):
        label=label if isinstance(label,tuple) else (label,)
        baseline=float(g.ref_error.mean()); candidate=float(g.absolute_error.mean())
        rows.append(dict(zip(by,label),observations=len(g),
                         reference_mae=baseline, challenger_mae=candidate,
                         gain_washes_per_hour=baseline-candidate,
                         skill_percent=100*(1-candidate/baseline) if baseline>0 else np.nan,
                         fraction_hours_better=float((g.absolute_error < g.ref_error).mean())))
    return pd.DataFrame(rows)


def make_reports(source: pd.DataFrame, *, reference: str='weather_median') -> dict[str,pd.DataFrame]:
    df=validate_predictions(source,reference=reference)
    reports={
        'overall_metrics.csv':summarize(df,['model']).sort_values('mae'),
        'by_location.csv':summarize(df,['model','location_id']),
        'by_weather_regime.csv':summarize(df,['model','weather_regime']),
        'by_location_and_regime.csv':summarize(df,['model','location_id','weather_regime']),
        'by_issuance.csv':summarize(df,['model','issue_date']),
        'by_location_and_issuance.csv':summarize(df,['model','location_id','issue_date']),
        'by_lead_day.csv':summarize(df,['model','lead_day']),
        'skill_vs_weather_median.csv':paired_skill(df,[],reference),
        'skill_vs_weather_by_location.csv':paired_skill(df,['location_id'],reference),
        'skill_vs_weather_by_regime.csv':paired_skill(df,['weather_regime'],reference),
        'skill_vs_weather_by_issuance.csv':paired_skill(df,['issue_date'],reference),
        'skill_vs_weather_by_location_and_regime.csv':paired_skill(df,['location_id','weather_regime'],reference),
    }
    if 'inferred_zero' in df:
        reports['by_inferred_zero.csv']=summarize(df,['model','inferred_zero'])
        reports['excluding_inferred_zeros.csv']=summarize(df.loc[~df.inferred_zero],['model'])
    if 'store_age_group' in df:
        reports['by_store_age.csv']=summarize(df,['model','store_age_group'])
    if 'rd_daily_days_28d' in df:
        df['history_28d_group']=pd.cut(pd.to_numeric(df.rd_daily_days_28d,errors='coerce'),
             bins=[-1,0,7,21,28],labels=['0 valid days','1-7 valid days','8-21 valid days','22-28 valid days'])
        reports['by_history_support.csv']=summarize(df,['model','history_28d_group'])
    store_mae=reports['by_location.csv']
    macro=(store_mae.groupby('model',sort=True).agg(
        locations=('location_id','nunique'),macro_store_mae=('mae','mean'),
        median_store_mae=('mae','median'),worst_store_mae=('mae','max'),
        max_abs_store_bias=('signed_bias',lambda x: float(np.max(np.abs(x))))).reset_index())
    reports['store_macro_metrics.csv']=macro.sort_values('macro_store_mae')
    # The most difficult forecast store/dates. These groups are NOT independent
    # storm-event replicates; the same target dates may occur at multiple issues.
    worst=reports['by_location_and_issuance.csv'].sort_values('mae',ascending=False)
    reports['worst_location_issuances.csv']=worst.groupby('model',sort=False).head(20)
    rainy=df[df.weather_regime.eq('Rain forecast for target hour')]
    if not rainy.empty:
        reports['rainy_store_day.csv']=summarize(rainy,['model','location_id','issue_date','local_date'])
        reports['rainy_issue.csv']=summarize(rainy,['model','issue_date'])
        reports['skill_vs_weather_rainy_store_day.csv']=paired_skill(rainy,['location_id','issue_date','local_date'],reference)
    # Evaluate summed daily volume ONLY for store/issue/days with an entire
    # expected operating grid (11 hours Mon-Sat, 10 on Sunday).
    dated=df.copy()
    dates=pd.to_datetime(dated.local_date)
    dated['expected_hours']=np.where(dates.dt.dayofweek.eq(6),10,11)
    daily=(dated.groupby(['model','issued_at','location_id','local_date'],as_index=False)
          .agg(carwashes=('carwashes','sum'),prediction=('prediction','sum'),
               hours=('timestamp','nunique'),expected_hours=('expected_hours','first')))
    daily=daily[daily.hours.eq(daily.expected_hours)].copy()
    daily['absolute_error']=np.abs(daily.carwashes-daily.prediction)
    daily['signed_error']=daily.prediction-daily.carwashes
    reports['daily_total_metrics.csv']=summarize(daily,['model'])
    reports['daily_total_by_location.csv']=summarize(daily,['model','location_id'])
    return reports


def make_parser():
    p=argparse.ArgumentParser(description='Evaluate paired WashCast forecasts and weather reliability')
    p.add_argument('--predictions',required=True)
    p.add_argument('--reference',default='weather_median')
    p.add_argument('--output',default='forecast_reliability_report')
    return p


def main(argv=None):
    a=make_parser().parse_args(argv)
    source=pd.read_csv(a.predictions,dtype={'location_id':str})
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    reports=make_reports(source,reference=a.reference)
    for name,table in reports.items():
        table.to_csv(out/name,index=False)
    (out/'README_REPORTS.txt').write_text(
        'Reports use archived *forecast* rainfall regimes, NOT realized weather.\\n'
        'Forecast issuances, nearby locations and repeated target hours are correlated.\\n'
        'Store-day rainy episodes are descriptive groupings, NOT independent storms.\\n'
        'Positive skill_percent means lower MAE than the weather-adjusted median.\\n'
        'Store-age is measured from first labeled valid history, not necessarily the real opening date.\\n',
        encoding='utf-8')
    print(reports['overall_metrics.csv'].to_string(index=False),flush=True)
    print('Reports:',out.resolve(),flush=True)

if __name__=='__main__':
    main()
