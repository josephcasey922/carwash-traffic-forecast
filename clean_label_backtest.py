"""WashCast: strictly chronological label-quality training sensitivity experiment.

Does NOT correct labels or modify the existing forecasting system. Compares the
same LightGBM features and hyperparameters with alternative TRAINING masks.
Scores the same held-out forecast hours for every model, both unfiltered and in
quality-specific cohorts. Evaluation labels are never removed to improve the
headline score. Inferred zeros may be real: filters are hypotheses, not truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODES = ('original', 'drop_flagged_zeros', 'drop_flagged_days',
         'drop_flagged_and_network_days')
SCHEMA_VERSION = 1
KEY = ['issued_at', 'location_id', 'timestamp']


def synchronized_gap_days(unique_labels: pd.DataFrame, *, minimum_stores: int = 10,
                          fraction: float = .8, minimum_hours: int = 2) -> set[str]:
    """Exploratory signals only; shared zero hours can reflect real storms/closures."""
    h = unique_labels.copy()
    h['local_date'] = pd.to_datetime(h.local_date).dt.strftime('%Y-%m-%d')
    hour = (h.groupby(['local_date', 'hour'], observed=True)
        .agg(stores=('location_id','nunique'), zeros=('inferred_zero','sum')).reset_index())
    hour['trigger'] = (hour.stores.ge(minimum_stores)
                       & hour.zeros.ge(minimum_stores)
                       & hour.zeros.div(hour.stores).ge(fraction))
    count = hour.loc[hour.trigger].groupby('local_date', observed=True).size()
    return set(count[count.ge(minimum_hours)].index.astype(str))


def make_quality_tags(hist: pd.DataFrame, day_table: pd.DataFrame,
                      network_dates: set[str]) -> pd.DataFrame:
    """Unique, auditable (store, local calendar date) classification."""
    days = day_table[['location_id','local_date','quality_tier','inferred_hours',
                     'missing_scheduled_hours']].copy()
    days.local_date = pd.to_datetime(days.local_date).dt.strftime('%Y-%m-%d')
    days['flagged_store_day'] = days.quality_tier.isin(['C_4plus_inferred', 'D_incomplete'])
    days['synchronized_gap_day'] = days.local_date.isin(network_dates)
    if days.duplicated(['location_id','local_date']).any():
        raise ValueError('Duplicate store-day quality classification')
    return days


def attach_tags(frame: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    if 'inferred_zero' not in frame:
        raise ValueError('Missing inferred_zero provenance; cannot filter safely')
    result = frame.copy()
    result['local_date'] = pd.to_datetime(result.local_date).dt.strftime('%Y-%m-%d')
    result = result.merge(tags, on=['location_id','local_date'], how='left',
                          validate='many_to_one', indicator=True)
    if result._merge.ne('both').any() or result.quality_tier.isna().any():
        raise ValueError('Unclassified target days in training examples')
    result = result.drop(columns='_merge')
    inferred = result.inferred_zero
    if inferred.isna().any():
        raise ValueError('Inferred zero flag contains missing values')
    if not pd.api.types.is_bool_dtype(inferred):
        s = inferred.astype('string').str.strip().str.lower()
        if (~s.isin(['true','false'])).any():
            raise ValueError('Invalid inferred_zero flag')
        result['inferred_zero'] = s.eq('true')
    if ((result.inferred_zero & result.carwashes.ne(0)) |
        ((~result.inferred_zero) & result.carwashes.eq(0))).any():
        raise ValueError('Inconsistent inferred-zero provenance')
    return result


def training_filter(train: pd.DataFrame, mode: str) -> pd.Series:
    if mode not in MODES:
        raise ValueError(f'Unknown mode {mode}')
    if mode == 'original':
        return pd.Series(True, index=train.index)
    if mode == 'drop_flagged_zeros':
        # Retains recorded positive hours even on suspicious store-days.
        return ~(train.flagged_store_day & train.inferred_zero)
    if mode == 'drop_flagged_days':
        return ~train.flagged_store_day
    return ~(train.flagged_store_day | train.synchronized_gap_day)


def classify_eval(df: pd.DataFrame) -> pd.Series:
    return np.select([
        df.flagged_store_day,
        df.synchronized_gap_day,
        df.inferred_zero],
        ['flagged_store_day', 'synchronized_gap', 'other_inferred_zero'],
        default='other_recorded',
    )


def metrics(group: pd.DataFrame) -> dict:
    d = group.prediction.to_numpy(float) - group.carwashes.to_numpy(float)
    a = np.abs(d)
    y = group.carwashes.to_numpy(float)
    # Existing business thresholds, used ONLY for post-hoc evaluation.
    rounded = np.maximum(0, np.rint(group.prediction.to_numpy(float)))
    def bucket(v):
        return np.select([v<=17, v<=40, v<=60], [0, 1, 2], default=3)
    pb, ab = bucket(rounded), bucket(y)
    return dict(hours=int(len(group)), actual_avg=float(np.mean(y)),
                mae=float(np.mean(a)), median_ae=float(np.median(a)),
                wape_percent=float(100*a.sum()/y.sum()) if y.sum()>0 else None,
                signed_bias=float(np.mean(d)),
                bucket_agreement_percent=float(100*np.mean(pb==ab)),
                bucket_under_percent=float(100*np.mean(pb<ab)),
                p90_ae=float(np.quantile(a,.90)))


def report_predictions(predictions: pd.DataFrame, folder:Path) -> None:
    p = predictions.loc[predictions.lead_day.eq(1)].copy()
    if p.empty:raise ValueError('No next-day forecasts')
    p['quality_cohort'] = classify_eval(p)
    p['week'] = pd.to_datetime(p.local_date).dt.to_period('W').astype(str)
    cohorts = {
       'official_all': pd.Series(True,index=p.index),
       'exclude_4plus_store_days': ~p.flagged_store_day,
       'fully_recorded_days': p.quality_tier.eq('A_no_inferred'),
       'recorded_hours_only': ~p.inferred_zero,
       'network_gap_days_only': p.synchronized_gap_day,
       'flagged_store_days_only': p.flagged_store_day,
    }
    overall=[]; by_store=[]; by_issue=[]; by_week=[]; by_quality=[]
    for cohort, mask in cohorts.items():
        subset=p.loc[mask]
        for mode,g in subset.groupby('model',observed=True):
            if not len(g): continue
            overall.append(dict(cohort=cohort,model=mode,**metrics(g)))
            if cohort in ('official_all','exclude_4plus_store_days','fully_recorded_days'):
                for store, s in g.groupby('location_id',observed=True):
                    by_store.append(dict(cohort=cohort,model=mode,location_id=store,**metrics(s)))
    for (model, issue), g in p.groupby(['model','issued_at'],observed=True):
        by_issue.append(dict(model=model,issued_at=str(issue),**metrics(g)))
    for (model,week), g in p.groupby(['model','week'],observed=True):
        by_week.append(dict(model=model,week=week,**metrics(g)))
    for (model,cohort),g in p.groupby(['model','quality_cohort'],observed=True):
        by_quality.append(dict(model=model,quality_cohort=cohort,**metrics(g)))
    for name, rows in [('day1_metrics_by_cohort.csv',overall),
                       ('day1_metrics_by_store.csv',by_store),
                       ('day1_metrics_by_issuance.csv',by_issue),
                       ('day1_metrics_by_week.csv',by_week),
                       ('day1_metrics_by_quality_type.csv',by_quality)]:
        pd.DataFrame(rows).to_csv(folder/name,index=False)
    print('\nPaired Day-1 accuracy (same target hours for every candidate):')
    print(pd.DataFrame(overall).query("cohort=='official_all'")[
       ['model','hours','mae','median_ae','wape_percent','signed_bias',
        'bucket_agreement_percent']].sort_values('mae').to_string(index=False))
    print('\nDay-1 on store-days with no inferred zeros:')
    print(pd.DataFrame(overall).query("cohort=='fully_recorded_days'")[
       ['model','hours','mae','wape_percent','signed_bias']].sort_values('mae').to_string(index=False))


def fingerprint(path):
    st=Path(path).stat()
    return {'path':str(Path(path).resolve()),'size':st.st_size,
            'modified_ns':st.st_mtime_ns}


def save_csv(path:Path,df:pd.DataFrame):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial')
    df.to_csv(tmp,index=False)
    tmp.replace(path)


def run(args):
    # Defer project imports so data-quality functions remain independently testable.
    import backtest
    import demand_feature_ablation as ab
    import demand_model_benchmark as orig
    import experiments
    import model_tuning as tuning
    import storm_loss_experiment as sl
    import washcast_label_audit as qa

    modes=tuple(x.strip() for x in args.modes.split(',') if x.strip())
    if len(set(modes))!=len(modes) or not modes or any(x not in MODES for x in modes):
        raise ValueError('Invalid or duplicated --modes')
    if 'original' not in modes:
        raise ValueError('Include original to compare paired unchanged LightGBM')
    hist,_,_=qa.load_unique_labels(Path(args.examples))
    days=qa.make_days(hist)
    network=synchronized_gap_days(hist)
    tags=make_quality_tags(hist,days,network)
    source=orig.load_data(args.examples,args.storm_features,args.recent_demand_features)
    df=attach_tags(source,tags)
    issues=tuning.select_validation_issues(df,args.validation_from,args.validation_to,
                                           args.validation_origins)
    params=orig._selected_parameters(args.selected_configs)
    folder=Path(args.output); folder.mkdir(parents=True,exist_ok=True)
    manifest=dict(version=SCHEMA_VERSION,examples=fingerprint(args.examples),
       storm=fingerprint(args.storm_features),recent=fingerprint(args.recent_demand_features),
       selected_configurations=fingerprint(args.selected_configs) if args.selected_configs else None,
       modes=modes,issuances=[str(s) for s in issues],window_days=args.window_days,
       threads=args.threads,notes='Development only. Filtering ONLY training targets. '
       'The future target quality is NEVER used to build predictions. Recent-demand '
       'features are unchanged; this isolates training-label selection alone.')
    path=folder/'experiment_manifest.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=json.loads(json.dumps(manifest)):
            raise ValueError('Output folder contains a different experiment; use a new --output')
    else:path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    tags.to_csv(folder/'store_day_quality_used.csv',index=False)
    pd.DataFrame({'local_date': sorted(network)}).to_csv(folder/'network_gap_dates_review.csv',index=False)
    all_paths=[];counts=[]
    for issue in issues:
        train,test=backtest.fold_sets(df,issue,args.window_days)
        if len(train)<200 or train.issued_at.nunique()<8 or test.empty:
            raise ValueError(f'Not enough data at {issue}')
        if train.timestamp.ge(issue).any() or train.issued_at.ge(issue).any():
            raise RuntimeError('Future labels used in training')
        test=experiments.add_regimes(test)
        issue_string=issue.strftime('%Y%m%dT%H%M%S%z')
        for mode in modes:
            dest=folder/'fold_predictions'/issue_string/(mode+'.csv')
            all_paths.append(dest)
            keep=training_filter(train,mode)
            filtered=train.loc[keep].copy()
            if len(filtered)<200 or filtered.issued_at.nunique()<8:
                raise ValueError(f'Filter removes too much training data: {mode} {issue}')
            counts.append(dict(issued_at=str(issue),mode=mode,
                original_training_rows=len(train),training_rows_kept=len(filtered),
                unique_target_hours_removed=int(train.loc[~keep,['location_id','timestamp']]
                   .drop_duplicates().shape[0]),removed_inferred_zero_rows=int(
                   train.loc[~keep,'inferred_zero'].sum())))
            if dest.exists():
                print(f'{issue.date()} {mode} already complete',flush=True)
                continue
            proc=ab.make_transformer('lgbm_stable_recent')
            x=proc.fit_transform(ab.make_inputs(filtered,'lgbm_stable_recent'))
            xt=proc.transform(ab.make_inputs(test,'lgbm_stable_recent'))
            names=proc.get_feature_names_out()
            x=pd.DataFrame(x,columns=names)
            xt=pd.DataFrame(xt,columns=names)
            estimator=sl.make_estimator('lightgbm','poisson',threads=args.threads,
                                         overrides=params.get('lightgbm',{}))
            estimator.fit(x,filtered.carwashes.to_numpy(float))
            pred=np.asarray(sl.predict(estimator,'lightgbm','poisson',xt),float).ravel()
            if not np.isfinite(pred).all() or (pred<0).any() or len(pred)!=len(test):
                raise RuntimeError('Invalid model output')
            columns=['issued_at','location_id','timestamp','local_date','hour','dow','lead_day',
                     'carwashes','inferred_zero','quality_tier','flagged_store_day',
                     'synchronized_gap_day','precipitation','rain_prev72h','weather_regime']
            part=test[columns].copy()
            part['model']=mode
            part['prediction']=np.round(pred,6)
            save_csv(dest,part)
            day1=part.loc[part.lead_day.eq(1)]
            print(f'{issue.date()} {mode:<30} Day-1 MAE={np.mean(np.abs(day1.prediction-day1.carwashes)):.3f}'
                  f' train={len(filtered)}',flush=True)
    if any(not p.exists() for p in all_paths):
        raise RuntimeError('Some prediction folds are missing')
    p=pd.concat((pd.read_csv(path,dtype={'location_id':str}) for path in all_paths),
                ignore_index=True)
    # Validate each model scored on precisely the same forecast instances.
    baseline=p.loc[p.model.eq('original')][KEY+['lead_day','carwashes']].sort_values(KEY).reset_index(drop=True)
    for mode in modes:
        other=p.loc[p.model.eq(mode)][KEY+['lead_day','carwashes']].sort_values(KEY).reset_index(drop=True)
        if not other.equals(baseline):
            raise RuntimeError(f'Prediction evaluation targets not paired for {mode}')
    save_csv(folder/'all_predictions.csv',p)
    pd.DataFrame(counts).to_csv(folder/'train_rows_by_fold.csv',index=False)
    report_predictions(p,folder)
    print('\nAll results saved:',folder.resolve())


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--examples',required=True)
    p.add_argument('--storm-features',required=True)
    p.add_argument('--recent-demand-features',required=True)
    p.add_argument('--selected-configs',default=None)
    p.add_argument('--validation-from',default='2026-05-01')
    p.add_argument('--validation-to',default='2026-07-31')
    p.add_argument('--validation-origins',type=int,default=18)
    p.add_argument('--window-days',type=int,default=365)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--modes',default=','.join(MODES))
    p.add_argument('--output',default='clean_label_backtest_may_july')
    return p

if __name__ == '__main__':
    run(parser().parse_args())
