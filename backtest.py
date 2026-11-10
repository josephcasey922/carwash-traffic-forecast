"""Offline rolling-origin backtest of saved, already-joined as-of examples.

This NEVER calls any API or accesses user traffic input. Each fold refits the
requested models on labels observable before that historical forecast issuance.
Features in input examples must have been built with archived issued weather.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import asof  # adds `lead_day` to the shared feature definition
import washcast as core


NEEDED = {'issued_at','timestamp','location_id','local_date','hour','dow',
          'lead_day','carwashes',*core.CAT,*core.NUM}


def load_examples(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={'location_id': 'str'})
    missing = NEEDED - set(df.columns)
    if missing:
        raise ValueError(f'Examples CSV missing required fields: {sorted(missing)}')
    if df.empty:
        raise ValueError('No labeled examples in CSV')
    df['issued_at'] = pd.to_datetime(df.issued_at, utc=True, errors='raise')
    df['timestamp'] = pd.to_datetime(df.timestamp, utc=True, errors='raise')
    df['carwashes'] = pd.to_numeric(df.carwashes, errors='raise')
    if df['carwashes'].isna().any() or (df.carwashes < 0).any():
        raise ValueError('Invalid actual wash counts')
    df['location_id'] = df.location_id.astype(str)
    df['lead_day'] = pd.to_numeric(df.lead_day, errors='raise').astype(int)
    if df.duplicated(['issued_at','location_id','timestamp']).any():
        raise ValueError('Duplicate (issued_at, location_id, timestamp) rows')
    if (df['timestamp'] <= df['issued_at']).any():
        raise ValueError('Target timestamp must be strictly later than forecast issuance')
    if 'inferred_zero' in df:
        df['inferred_zero'] = df.inferred_zero.astype('string').str.lower().eq('true')
    return df.sort_values(['issued_at','location_id','timestamp']).reset_index(drop=True)


def fold_sets(df: pd.DataFrame, issue: pd.Timestamp, window_days: int):
    cutoff = issue - pd.Timedelta(days=window_days)
    # IMPORTANT: Both issuance and actual TARGET must be before current issue.
    # This prevents fitting a model using future labels from an earlier forecast.
    train = df[(df.issued_at < issue)&(df.timestamp < issue)&(df.issued_at >= cutoff)]
    test = df[df.issued_at == issue]
    if len(train) and train.timestamp.max() >= issue:
        raise RuntimeError('Temporal data leak detected')
    return train.copy(),test.copy()


def diagnostics(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows=[]
    for key, group in frame.groupby(group_cols,dropna=False,sort=True):
        key = key if isinstance(key,tuple) else (key,)
        entry = dict(zip(group_cols,key))
        entry.update(core.model_metrics(group.carwashes.to_numpy(),group.prediction.to_numpy()))
        entry['signed_bias'] = round(float(np.mean(group.prediction-group.carwashes)),3)
        rows.append(entry)
    return pd.DataFrame(rows)


def run_backtest(df: pd.DataFrame, models: list[str], test_origins: int,
                 window_days: int, min_train_rows: int = 200,
                 min_train_origins: int = 8):
    if not models or any(m not in core.ALL_MODELS for m in models) or len(models)!=len(set(models)):
        raise ValueError(f'Choose one or more unique model names from {core.ALL_MODELS}')
    if not 1 <= test_origins < df.issued_at.nunique():
        raise ValueError('Number of test origins must be >= 1 and less than number of available origins')
    if window_days <= 0:
        raise ValueError('window_days must be positive')
    origins = sorted(df.issued_at.unique())[-test_origins:]
    pred_parts=[]; fold_status=[]
    for issue in origins:
        train,test=fold_sets(df, issue,window_days)
        unique_train=train.issued_at.nunique()
        if len(train)<min_train_rows or unique_train<min_train_origins:
            raise ValueError(f'Insufficient data before {issue}: {len(train)} rows and {unique_train} issuance dates; '
                             'use fewer test origins, a larger window, or a smaller minimum.')
        if test.empty:
            raise ValueError(f'Missing test data for {issue}')
        if (train.issued_at >= issue).any() or (train.timestamp >= issue).any():
            raise RuntimeError('Backtest included future observations in the training fold')
        print(f'Issuance {issue}: fitting on {len(train):,} rows from {unique_train} origins; '
              f'checking {len(test):,} unseen predictions',flush=True)
        fold_status.append({'issued_at':str(issue),'n_train_rows':len(train),
                            'n_train_origins':int(unique_train),'n_test_rows':len(test),
                            'max_training_target':str(train.timestamp.max())})
        basic=['issued_at','timestamp','local_date','location_id','hour','dow','lead_day','carwashes']
        if 'inferred_zero' in test:
            basic.append('inferred_zero')
        for model_name in models:
            model=core.fit_model(model_name,train)
            predicted=core.predict_model(model_name,model,test)
            sample=test[basic].copy()
            sample['model']=model_name
            sample['prediction']=np.round(predicted,4)
            sample['absolute_error']=abs(sample.carwashes-sample.prediction)
            sample['signed_error']=sample.prediction-sample.carwashes
            pred_parts.append(sample)
    predictions=pd.concat(pred_parts,ignore_index=True)
    predictions['issue_date'] = predictions.issued_at.dt.strftime('%Y-%m-%d')
    diagnostics_files = {
        'overall_metrics.csv':diagnostics(predictions,['model']).sort_values('mae'),
        'by_issuance.csv':diagnostics(predictions,['model','issue_date']),
        'by_lead_day.csv':diagnostics(predictions,['model','lead_day']),
        'by_issue_and_lead.csv':diagnostics(predictions,['model','issue_date','lead_day']),
        'by_location.csv':diagnostics(predictions,['model','location_id']),
        'by_weekday.csv':diagnostics(predictions,['model','dow']),
    }
    if 'inferred_zero' in predictions:
        diagnostics_files['by_inferred_zero.csv'] = diagnostics(predictions,['model','inferred_zero'])
    return predictions,diagnostics_files,fold_status


def make_parser():
    parser = argparse.ArgumentParser(description='Fully offline rolling-origin evaluation (no API calls)')
    parser.add_argument('--examples',required=True,help='Path to existing asof_training_examples.csv')
    parser.add_argument('--test-origins',type=int,default=8,help='Evaluate the last N issued historical 7-day forecasts')
    parser.add_argument('--window-days',type=int,default=180,help='Rolling number of days of past issued forecast examples used per fold')
    parser.add_argument('--models',default='baseline,boosted,mlp')
    parser.add_argument('--min-train-rows',type=int,default=200)
    parser.add_argument('--min-train-origins',type=int,default=8)
    parser.add_argument('--output',default='rolling_backtest')
    return parser


def main(argv=None):
    args=make_parser().parse_args(argv)
    df=load_examples(args.examples)
    models=[s.strip() for s in args.models.split(',') if s.strip()]
    predictions,metrics,fold_status=run_backtest(df,models,args.test_origins,args.window_days,
                      args.min_train_rows,args.min_train_origins)
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    predictions.to_csv(dest/'backtest_predictions.csv',index=False)
    for name,table in metrics.items():
        table.to_csv(dest/name,index=False)
    record={'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'source_examples':str(Path(args.examples).resolve()),
            'source_rows':int(len(df)),
            'note':'Evaluation only: all models refitted separately for each forecast origin, no model artifacts saved, no APIs called',
            'models':models,'window_days':args.window_days,'test_origins':args.test_origins,
            'issuance_details':fold_status,'limitations':[
                'Weekly forecast origins alone confound forecast lead day with day-of-week.',
                'If these same results are used for model selection, they are validation, not a final independent test.',
                'Missing-ticket hours may be inferred zeros; inspect closures and outage periods.',
                'Comparison with other windows should use identical issuance dates and test-row identifiers.'
            ]}
    (dest/'backtest_info.json').write_text(json.dumps(record,indent=2,default=str))
    print('\nBacktest performance:')
    print(metrics['overall_metrics.csv'].to_string(index=False))
    print(f'\nResults saved in {dest.resolve()}',flush=True)

if __name__=='__main__':
    main()
