"""Read-only source-label audit for WashCast hourly count forecasting.

No model fitting, no imputation, and NO changes to training examples. Separates
observed positive transaction hours from inferred zero labels and groups
suspicious labels by store-day. Optional paired Day-1 forecast sensitivity
analysis never replaces the unfiltered official score.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ['location_id', 'timestamp']


def parse_bool(s: pd.Series) -> pd.Series:
    return s.astype('string').str.lower().str.strip().isin(['true', '1', 'yes', 'y'])


def longest_run(hours: np.ndarray) -> int:
    if len(hours) == 0:
        return 0
    arr = np.unique(np.sort(hours.astype(int)))
    longest = current = 1
    for a,b in zip(arr[:-1],arr[1:]):
        current = current+1 if b == a+1 else 1
        longest=max(longest,current)
    return longest


def load_unique_labels(path: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    frame=pd.read_csv(path, dtype={'location_id':str})
    required={'location_id','timestamp','local_date','hour','carwashes','inferred_zero'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Example CSV missing {sorted(required-set(frame.columns))}')
    frame['timestamp']=pd.to_datetime(frame.timestamp, utc=True, errors='raise')
    frame['local_date']=pd.to_datetime(frame.local_date, errors='raise').dt.normalize()
    frame['carwashes']=pd.to_numeric(frame.carwashes,errors='raise')
    frame['hour']=pd.to_numeric(frame.hour, errors='raise').astype(int)
    frame['inferred_zero']=parse_bool(frame.inferred_zero)
    if (frame.carwashes.isna()|~np.isfinite(frame.carwashes)|(frame.carwashes<0)).any():
        raise ValueError('Invalid source labels')
    contradictions={}
    g=frame.groupby(KEY,observed=True, sort=False)
    for col in ['carwashes','local_date','hour','inferred_zero']:
        contradictions[col]=int(g[col].nunique(dropna=False).gt(1).sum())
    if any(contradictions.values()):
        raise ValueError(f'Contradictory repeated labels: {contradictions}')
    if ((frame.inferred_zero & frame.carwashes.ne(0)) | ((~frame.inferred_zero) & frame.carwashes.eq(0))).any():
        raise ValueError('Inconsistent zero provenance: inferred_zero must match zero counts in this export')
    hist=frame.drop_duplicates(KEY).copy()
    hist=hist.sort_values(['location_id','timestamp']).reset_index(drop=True)
    d={
      'all_example_rows':int(len(frame)),
      'unique_store_target_hours':int(len(hist)),
      'repeated_example_rows':int(len(frame)-len(hist)),
      'unique_inferred_zero_hours':int(hist.inferred_zero.sum()),
      'inferred_zero_fraction_percent':round(float(100*hist.inferred_zero.mean()),3),
      'number_of_stores':int(hist.location_id.nunique()),
    }
    return hist,d,frame


def make_days(hist:pd.DataFrame, threshold:int=3) -> pd.DataFrame:
    g=hist.groupby(['location_id','local_date'],sort=True,observed=True)
    days=g.agg(hours=('hour','nunique'),inferred_hours=('inferred_zero','sum'),
               observed_hours=('inferred_zero',lambda s:int((~s).sum())),
               observed_washes=('carwashes','sum'),min_hour=('hour','min'),max_hour=('hour','max')).reset_index()
    days['expected_hours']=np.where(days.local_date.dt.dayofweek.eq(6),10,11)
    days['missing_scheduled_hours']=days.expected_hours-days.hours
    days['longest_inferred_run']=[longest_run(z.loc[z.inferred_zero,'hour'].to_numpy())
                                  for _,z in g]
    days['zero_fraction']=days.inferred_hours / days.hours
    days['quality_tier']=np.select([
        days.missing_scheduled_hours.ne(0),
        days.inferred_hours.gt(threshold),
        days.inferred_hours.gt(0)],
        ['D_incomplete','C_4plus_inferred','B_1to3_inferred'], default='A_no_inferred')
    days['review_priority']=np.select([
        days.missing_scheduled_hours.ne(0),
        days.inferred_hours.ge(5),
        days.inferred_hours.gt(threshold),
        days.longest_inferred_run.ge(3)],
        ['high_incomplete','high_5plus','review_4plus','review_long_run'],default='routine')
    return days


def missing_store_days(days:pd.DataFrame) -> pd.DataFrame:
    pieces=[]
    for site,g in days.groupby('location_id',sort=True):
        expected=pd.date_range(g.local_date.min(),g.local_date.max(),freq='D')
        missing=expected.difference(pd.DatetimeIndex(g.local_date))
        if len(missing):
            pieces.append(pd.DataFrame({'location_id':site,'local_date':missing,
                                        'interpretation':'unknown_or_closed_not_zero'}))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=['location_id','local_date','interpretation'])


def raw_crosscheck(raw_path:Path,hist:pd.DataFrame) -> tuple[pd.DataFrame,dict]:
    raw=pd.read_csv(raw_path,dtype={'location_code_id':str,'location_id':str})
    for src,dest in [('hour','timestamp'),('location_code_id','location_id'),('num_carwashes','carwashes')]:
        if src in raw.columns:
            if dest in raw: raise ValueError(f'Raw CSV has both {src} and {dest}')
            raw=raw.rename(columns={src:dest})
    if not {'timestamp','location_id','carwashes'}.issubset(raw):
        raise ValueError('Raw traffic missing timestamp, location_id or carwashes')
    raw.timestamp=pd.to_datetime(raw.timestamp,utc=True,errors='raise').dt.floor('h')
    raw.carwashes=pd.to_numeric(raw.carwashes,errors='raise')
    byhour=raw.groupby(KEY,as_index=False).agg(raw_washes=('carwashes','sum'),n_source_rows=('carwashes','size'))
    merged=hist.merge(byhour,on=KEY,how='left',validate='one_to_one')
    disagree=(merged.raw_washes.notna() & merged.carwashes.ne(merged.raw_washes))
    unexpected_zero=(merged.inferred_zero & merged.raw_washes.notna())
    missing_source=(~merged.inferred_zero & merged.raw_washes.isna())
    issues=merged.loc[disagree|unexpected_zero|missing_source,KEY+['local_date','hour','carwashes','inferred_zero','raw_washes']].copy()
    details={'raw_source_rows':int(len(raw)),'unique_raw_store_hours':int(len(byhour)),
             'raw_explicit_zero_hours':int((byhour.raw_washes==0).sum()),
             'label_count_mismatches':int(disagree.sum()),
             'labels_inferred_but_in_raw':int(unexpected_zero.sum()),
             'labels_marked_recorded_but_not_in_raw':int(missing_source.sum())}
    return issues,details


def safe_wape(g):
    return 100*float(g.abs_error.sum())/float(g.carwashes.sum()) if g.carwashes.sum()>0 else np.nan


def eval_metrics(g):
    return pd.Series(dict(hours=len(g),avg_actual=float(g.carwashes.mean()),
            mae=float(g.abs_error.mean()),median_ae=float(g.abs_error.median()),
            wape_percent=safe_wape(g),signed_bias=float(g.error.mean()),
            inferred_hours=int(g.inferred_zero.sum()),
            predicted_61plus=int((g.prediction>=60.5).sum()),actual_61plus=int((g.carwashes>=61).sum())))


def optional_prediction_reports(predictions_path:Path,days:pd.DataFrame,output:Path,
                                model:str,reference:str):
    p=pd.read_csv(predictions_path,dtype={'location_id':str})
    required={'location_id','local_date','hour','lead_day','model','carwashes','prediction'}
    if not required.issubset(p):
        raise ValueError(f'Prediction CSV missing {sorted(required-set(p))}')
    p=p[(p.lead_day==1)&p.model.isin([model,reference])].copy()
    if p.empty: raise ValueError('No Day-1 predictions for requested models')
    p['local_date']=pd.to_datetime(p.local_date).dt.normalize()
    p['hour']=pd.to_numeric(p.hour,errors='raise').astype(int)
    p['carwashes']=pd.to_numeric(p.carwashes,errors='raise')
    p['prediction']=pd.to_numeric(p.prediction,errors='raise')
    if p.duplicated(['model','location_id','local_date','hour']).any():
        raise ValueError('Duplicate Day-1 model/store/day/hour; align issuance before scoring')
    p=p.merge(days[['location_id','local_date','quality_tier','review_priority','inferred_hours','longest_inferred_run']],
              on=['location_id','local_date'],how='left',validate='many_to_one')
    if p.quality_tier.isna().any():
        raise ValueError('Some predicted Day-1 labels are absent from source examples')
    p['inferred_zero']=False
    # Derived from hourly observed==zero: source has no recorded zero labels.
    p['inferred_zero']=p.carwashes.eq(0)
    p['error']=p.prediction-p.carwashes
    p['abs_error']=p.error.abs()
    # Scores are descriptive sensitivities, never alternate official model scores.
    cohorts={
      'official_all_days':lambda f:pd.Series(True,index=f.index),
      'no_flagged_4plus_days':lambda f:~f.quality_tier.isin(['C_4plus_inferred','D_incomplete']),
      'fully_observed_days_only':lambda f:f.quality_tier.eq('A_no_inferred'),
      'recorded_positive_hours_only':lambda f:~f.inferred_zero,
      'flagged_4plus_days_only':lambda f:f.quality_tier.eq('C_4plus_inferred'),
    }
    rows=[];store_rows=[]
    for label,fn in cohorts.items():
        sample=p.loc[fn(p)]
        for mdl,g in sample.groupby('model'):
            if g.empty:continue
            vals=eval_metrics(g).to_dict()
            rows.append(dict(cohort=label,model=mdl,**vals))
            if label in ['official_all_days','no_flagged_4plus_days','fully_observed_days_only']:
                for store,ss in g.groupby('location_id'):
                    store_rows.append(dict(cohort=label,model=mdl,location_id=store,**eval_metrics(ss).to_dict()))
    pd.DataFrame(rows).to_csv(output/'day1_quality_sensitivity.csv',index=False)
    pd.DataFrame(store_rows).to_csv(output/'day1_store_quality_sensitivity.csv',index=False)
    breakdown=[]
    for (mdl,quality),g in p.groupby(['model','quality_tier'],observed=True):
        breakdown.append(dict(model=mdl,quality_tier=quality,**eval_metrics(g).to_dict()))
    pd.DataFrame(breakdown).to_csv(output/'day1_by_label_quality.csv',index=False)
    return dict(prediction_hours=int(len(p)),models=sorted(p.model.unique().tolist()))


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    hist,info,all_examples=load_unique_labels(Path(args.examples))
    days=make_days(hist,args.max_inferred_per_day)
    missing=missing_store_days(days)
    days.to_csv(out/'store_day_label_quality.csv',index=False)
    missing.to_csv(out/'unobserved_store_days.csv',index=False)
    summary=(days.groupby('location_id',observed=True)
       .agg(days=('local_date','size'),inferred_hours=('inferred_hours','sum'),days_any_inferred=('inferred_hours',lambda x:int((x>0).sum())),
            flagged_days=('quality_tier',lambda x:int(x.isin(['C_4plus_inferred','D_incomplete']).sum())),
            severe_5plus_days=('inferred_hours',lambda x:int((x>=5).sum())),
            longest_gap=('longest_inferred_run','max'))
       .reset_index())
    summary.to_csv(out/'store_quality_summary.csv',index=False)
    hist_month=hist.copy();hist_month['month']=hist_month.local_date.dt.strftime('%Y-%m')
    month=(hist_month.groupby('month',observed=True).agg(hours=('hour','size'),inferred_hours=('inferred_zero','sum'),
                          avg_washes=('carwashes','mean'),total_washes=('carwashes','sum')).reset_index())
    month['inferred_percent']=100*month.inferred_hours/month.hours
    month.to_csv(out/'monthly_label_quality.csv',index=False)
    net=(days.groupby('local_date',observed=True).agg(stores=('location_id','nunique'),inferred_hours=('inferred_hours','sum'),
           stores_4plus=('inferred_hours',lambda x:int((x>=4).sum())),
           stores_any=('inferred_hours',lambda x:int((x>0).sum())),network_recorded_washes=('observed_washes','sum')).reset_index())
    net=net.sort_values(['inferred_hours','stores_4plus'],ascending=False)
    net.to_csv(out/'network_missingness_days.csv',index=False)
    count_by_day=days.groupby('quality_tier').size().to_dict()
    info.update({'unique_store_days':int(len(days)),
      'days_4plus_inferred':int((days.inferred_hours>args.max_inferred_per_day).sum()),
      'days_5plus_inferred':int((days.inferred_hours>=5).sum()),
      'unobserved_store_days_within_example_history':int(len(missing)),
      'quality_tier_counts':{str(k):int(v) for k,v in count_by_day.items()},
      'note':'Inferred zeros represent unknown data provenance, not proven missing traffic. Do not automatically delete storms or verified closures.'})
    if args.raw_traffic:
        issues,raw_info=raw_crosscheck(Path(args.raw_traffic),hist)
        issues.to_csv(out/'raw_vs_label_disagreements.csv',index=False)
        info['raw_crosscheck']=raw_info
    if args.predictions:
        info['day1_evaluation']=optional_prediction_reports(Path(args.predictions),days,out,args.model,args.reference)
    import json
    (out/'audit_summary.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print('Read-only audit complete:',out.resolve())
    print(json.dumps(info,indent=2))
    print('Worst network missingness days:')
    print(net.head(8).to_string(index=False))
    if args.predictions:
        print('Day-1 sensitivity:')
        print(pd.read_csv(out/'day1_quality_sensitivity.csv')[['cohort','model','hours','mae','wape_percent']].to_string(index=False))


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--examples',required=True,help='Your asof_training_examples.csv (read only)')
    a.add_argument('--raw-traffic',help='Original raw carwash_traffic.csv (optional)')
    a.add_argument('--predictions',help='Model benchmark all_predictions.csv (optional)')
    a.add_argument('--model',default='lgbm_stable_recent')
    a.add_argument('--reference',default='weather_median')
    a.add_argument('--max-inferred-per-day',type=int,default=3)
    a.add_argument('--output',default='washcast_label_audit')
    run(a.parse_args())

if __name__ == '__main__':
    main()
