"""Additional per-store, relative-scale, weather-consistency and cold-start reports.

Designed for existing WashCast all_predictions.csv, including previous runs.
WAPE = sum(|error|) / sum(actual); this equals MAE / mean actual within a
single group (where total actual > 0). Never use per-hour MAPE for zero counts.
Forecast-rain store-days and issuance dates are NOT independent storms.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import forecast_reliability as legacy

JOIN=['issued_at','location_id','timestamp']
RAINFALL='Rain forecast for target hour'


def _numerics(d: pd.DataFrame) -> pd.DataFrame:
    d=d.copy()
    if 'rd_daily_mean_28d' in d:
        d['rd_daily_mean_28d']=pd.to_numeric(d['rd_daily_mean_28d'],errors='coerce')
    if 'rd_store_age_days' in d:
        d['rd_store_age_days']=pd.to_numeric(d['rd_store_age_days'],errors='coerce')
    return d


def store_scale_metrics(df: pd.DataFrame, reference: str) -> pd.DataFrame:
    rows=[]
    ref=legacy.summarize(df[df.model.eq(reference)],['location_id']).set_index('location_id')
    for (model, site),part in df.groupby(['model','location_id'],sort=True):
        y=part.carwashes.to_numpy(dtype=float)
        err=part.absolute_error.to_numpy(dtype=float)
        bias=part.signed_error.to_numpy(dtype=float)
        mean_y=float(y.mean())
        mae=float(err.mean())
        base=float(ref.loc[site,'mae'])
        rows.append({
            'model':model,'location_id':site,'observations':len(part),
            'actual_mean_washes_per_hour':mean_y,
            'actual_median_washes_per_hour':float(np.median(y)),
            'mae_washes_per_hour':mae,
            'wape_percent':float(100*mae/mean_y) if mean_y>0 else np.nan,
            'p90_absolute_error_washes':float(np.quantile(err,0.9)),
            'p90_error_percent_of_store_average':float(100*np.quantile(err,0.9)/mean_y) if mean_y>0 else np.nan,
            'signed_bias_washes_per_hour':float(bias.mean()),
            'signed_bias_percent_of_store_average':float(100*bias.mean()/mean_y) if mean_y>0 else np.nan,
            'reference_mae':base,
            'skill_vs_reference_percent':float(100*(1-mae/base)) if base>0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values(['model','location_id']).reset_index(drop=True)


def _finite_aggregate(values, fn):
    values=np.asarray(values,dtype=float)
    good=values[np.isfinite(values)]
    return float(fn(good)) if len(good) else np.nan


def macro_relative(store: pd.DataFrame, reference: str) -> pd.DataFrame:
    rows=[]
    for model,part in store.groupby('model',sort=True):
        diff=part.skill_vs_reference_percent.to_numpy(dtype=float)
        bias=part.signed_bias_percent_of_store_average.to_numpy(dtype=float)
        wape=part.wape_percent.to_numpy(dtype=float)
        abs_bias=np.abs(bias)
        rows.append({
            'model':model,
            'locations':len(part),
            'macro_store_wape_percent':_finite_aggregate(wape,np.mean),
            'median_store_wape_percent':_finite_aggregate(wape,np.median),
            'worst_store_wape_percent':_finite_aggregate(wape,np.max),
            'median_store_skill_vs_reference_percent':_finite_aggregate(diff,np.median),
            'fraction_stores_beating_reference':float(np.mean(diff>1e-9)) if model!=reference else 0.,
            'max_abs_store_bias_percent_of_average':_finite_aggregate(abs_bias,np.max),
            'mean_store_mae':float(part.mae_washes_per_hour.mean()),
        })
    return pd.DataFrame(rows).sort_values('macro_store_wape_percent').reset_index(drop=True)


def _paired(df: pd.DataFrame, reference: str, group_cols: list[str]) -> pd.DataFrame:
    base=(df[df.model.eq(reference)][JOIN+['absolute_error']]
          .rename(columns={'absolute_error':'reference_absolute_error'}))
    rest=df[df.model.ne(reference)].merge(base,on=JOIN,how='left',validate='many_to_one')
    if rest.reference_absolute_error.isna().any():
        raise ValueError('Paired comparison has missing reference predictions')
    items=[]
    for keys,group in rest.groupby(['model']+group_cols,sort=True,observed=True,dropna=False):
        keys=keys if isinstance(keys,tuple) else (keys,)
        ref_mae=float(group.reference_absolute_error.mean())
        mae=float(group.absolute_error.mean())
        items.append({**dict(zip(['model']+group_cols,keys)),
                      'observations':len(group),
                      'reference_mae':ref_mae, 'challenger_mae':mae,
                      'gain_washes_per_hour':ref_mae-mae,
                      'skill_percent':100*(1-mae/ref_mae) if ref_mae>0 else np.nan,
                      'fraction_hours_better':float((group.absolute_error<group.reference_absolute_error).mean()),
                      })
    return pd.DataFrame(items)


def consistency_across_issuances(pairs: pd.DataFrame, *, min_hours: int) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    rows=[]
    for (model, regime), part in pairs.groupby(['model','weather_regime'],sort=True):
        substantial=part[part.observations>=min_hours]
        gains=substantial.gain_washes_per_hour.to_numpy(dtype=float)
        rows.append({
            'model':model,'weather_regime':regime,
            'issuances_with_regime':len(part),
            'issuances_with_at_least_min_hours':len(substantial),
            'minimum_hours_per_issuance':min_hours,
            'total_hours_in_eligible_issuances':int(substantial.observations.sum()),
            'fraction_issuances_better':float((gains>0).mean()) if len(gains) else np.nan,
            'median_issuance_gain':float(np.median(gains)) if len(gains) else np.nan,
            'p10_issuance_gain':float(np.quantile(gains,.1)) if len(gains) else np.nan,
            'worst_issuance_gain':float(gains.min()) if len(gains) else np.nan,
            'mean_issuance_gain':float(gains.mean()) if len(gains) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(['model','weather_regime']).reset_index(drop=True)


def volume_bands(df: pd.DataFrame) -> pd.DataFrame:
    # This *as-of* estimate comes from recent complete-day history; it is not
    # bucketed using the realized traffic of the target forecast hour.
    date=pd.to_datetime(df.local_date,errors='raise')
    hours=np.where(date.dt.dayofweek.eq(6),10.,11.)
    normal=df.rd_daily_mean_28d.to_numpy(dtype=float)/hours
    df=df.copy()
    df['estimated_typical_hourly_traffic']=normal
    levels=pd.cut(normal,[-np.inf,5.,15.,30.,np.inf],
                  labels=['0-5','5-15','15-30','>30'])
    df['historical_volume_band']=levels.astype(object)
    df['historical_volume_band']=df.historical_volume_band.fillna('unknown')
    return legacy.summarize(df,['model','historical_volume_band'])


def make_reports(source: pd.DataFrame, *, reference: str='weather_median',
                 comparison: str='lgbm_no_recent', min_weather_hours: int=10) -> dict[str,pd.DataFrame]:
    if min_weather_hours<=0:
        raise ValueError('min_weather_hours must be positive')
    df=_numerics(legacy.validate_predictions(source,reference=reference))
    store=store_scale_metrics(df,reference)
    reports={
        'store_relative_reliability.csv':store,
        'macro_relative_reliability.csv':macro_relative(store,reference),
        'weather_by_issuance_paired_vs_median.csv':_paired(df,reference,['issue_date','weather_regime']),
        'weather_by_store_and_issuance_vs_median.csv':_paired(df,reference,['location_id','issue_date','weather_regime']),
    }
    reports['weather_consistency_vs_median.csv']=consistency_across_issuances(
        reports['weather_by_issuance_paired_vs_median.csv'],min_hours=min_weather_hours)
    if comparison in set(df.model):
        reports['weather_by_issuance_paired_vs_no_recent.csv']=_paired(
            df,comparison,['issue_date','weather_regime'])
        reports['weather_consistency_vs_no_recent.csv']=consistency_across_issuances(
            reports['weather_by_issuance_paired_vs_no_recent.csv'],min_hours=min_weather_hours)
        reports['skill_by_location_vs_no_recent.csv']=_paired(df,comparison,['location_id'])
    if 'rd_daily_mean_28d' in df:
        reports['by_historical_volume_band.csv']=volume_bands(df)
    if 'store_age_group' in df:
        reports['by_observed_store_age.csv']=legacy.summarize(df,['model','store_age_group'])
        reports['skill_by_store_age_vs_median.csv']=_paired(df,reference,['store_age_group'])
    if 'inferred_zero' in df:
        reports['quality_sensitivity.csv']=legacy.summarize(df,['model','inferred_zero'])
        reports['excluding_inferred_zero_hours.csv']=legacy.summarize(df.loc[~df.inferred_zero],['model'])
    rainy=df.loc[df.weather_regime.eq(RAINFALL)].copy()
    if not rainy.empty:
        # Store-by-forecast-issuance-by-target-day. Not a physical/independent
        # weather event: nearby locations share weather and issues overlap.
        reports['rainy_store_day_paired_vs_median.csv']=_paired(
            rainy,reference,['location_id','issue_date','local_date'])
        reports['rainy_store_day_paired_vs_median.csv']=reports[
            'rainy_store_day_paired_vs_median.csv'].sort_values('gain_washes_per_hour')
    return reports


def main(argv=None):
    p=argparse.ArgumentParser(description='Relative-scale and weather consistency reports for WashCast forecasts')
    p.add_argument('--predictions',required=True)
    p.add_argument('--reference',default='weather_median')
    p.add_argument('--comparison',default='xgb_original')
    p.add_argument('--min-weather-hours',type=int,default=10)
    p.add_argument('--output',default='reliability_extended_report')
    a=p.parse_args(argv)
    data=pd.read_csv(a.predictions,dtype={'location_id':str})
    folder=Path(a.output);folder.mkdir(parents=True,exist_ok=True)
    reports=make_reports(data,reference=a.reference,comparison=a.comparison,
                         min_weather_hours=a.min_weather_hours)
    for name,table in reports.items():
        table.to_csv(folder/name,index=False)
    (folder/'README_REPORTS.txt').write_text(
        'WAPE by store = 100 * sum(|pred - actual|) / sum(actual). No per-hour MAPE is used.\n'
        'MAE and its 90th percentile are also scaled relative to the store mean actual count.\n'
        'Store-mean and skill scaling are descriptive evaluation statistics, NOT forecasting inputs.\n'
        'Only forecast rain is available here; these are NOT verified independent weather events.\n'
        'A date with multiple sites/lead days is not an independent observation.\n'
        'Store age is computed from first valid observed traffic, not confirmed official opening dates.\n'
        'Per-issue consistency requires a minimum number of regime rows; sparse dates are excluded.\n',
        encoding='utf-8')
    print(reports['macro_relative_reliability.csv'].to_string(index=False))
    print('Reports:',folder.resolve())


if __name__=='__main__':
    main()
