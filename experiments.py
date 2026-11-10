"""Offline paired weather ablation and chronologically tuned MLP rolling backtests.

Inputs are the existing as-of examples with archived forecasts. No network access.
Each test issue refits models using strictly earlier observable labels. MLP hyperparameters
and epoch count are selected on an *inner* earlier chronological validation split.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import backtest
import washcast as core

WEATHER_NUM = ('temperature_2m', 'precipitation', 'cloud_cover', 'rain_prev24h', 'rain_prev72h')
SEASON_NUM = ('season_sin', 'season_cos')
REGIME_ORDER = ['Rain forecast for target hour', 'Rain in previous 72h', 'Otherwise dry']


def numeric_cols(use_weather: bool) -> list[str]:
    return list(core.NUM) + list(SEASON_NUM) if use_weather else [name for name in core.NUM if name not in WEATHER_NUM] + list(SEASON_NUM)


def feature_columns(use_weather: bool) -> list[str]:
    return list(core.CAT) + numeric_cols(use_weather)


def make_transformer(use_weather: bool) -> ColumnTransformer:
    return ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), list(core.CAT)),
        ('num', Pipeline([('fill', SimpleImputer(strategy='median')),
                          ('scale', StandardScaler())]), numeric_cols(use_weather)),
    ], sparse_threshold=0)


def inputs(df: pd.DataFrame, use_weather: bool) -> pd.DataFrame:
    x = df[feature_columns(use_weather)].copy()
    for c in core.CAT:
        x[c] = x[c].astype(str)
    return x


def boost_model(train: pd.DataFrame, use_weather: bool):
    model = Pipeline([
        ('features', make_transformer(use_weather)),
        ('model', HistGradientBoostingRegressor(
            loss='poisson', max_iter=130, max_leaf_nodes=19,
            learning_rate=.07, l2_regularization=.5, random_state=42)),
    ])
    model.fit(inputs(train, use_weather), train.carwashes.to_numpy(dtype=float))
    return model


def parse_candidates(spec: str) -> list[tuple[tuple[int, ...], float]]:
    candidates = []
    for item in spec.split(','):
        if not item.strip():
            continue
        try:
            topology, penalty = item.strip().split(':')
            hidden = tuple(int(n) for n in topology.lower().split('x'))
            alpha = float(penalty)
        except (ValueError, TypeError) as exc:
            raise ValueError('MLP candidate format must be like 12x6:0.3,24x12:1.5') from exc
        if not hidden or any(n <= 0 or n > 256 for n in hidden) or not (0 <= alpha <= 1000):
            raise ValueError('Invalid MLP hidden sizes or alpha')
        candidates.append((hidden, alpha))
    if not candidates:
        raise ValueError('At least one MLP candidate is required')
    return candidates


def inner_split(train: pd.DataFrame, validation_issues: int):
    issues = sorted(train.issued_at.unique())
    if len(issues) < validation_issues + 3:
        raise ValueError('Not enough historical issuances for separate inner MLP validation; '
                         'increase training window, use fewer inner origins, or disable MLP')
    first_val_issue = issues[-validation_issues]
    # All inner model fitting uses labels that would have been observed before the
    # first inner validation issuance. No label from the inner validation horizon leaks.
    inner_train = train[(train.issued_at < first_val_issue) & (train.timestamp < first_val_issue)]
    inner_val = train[train.issued_at >= first_val_issue]
    if inner_train.empty or inner_val.empty:
        raise ValueError('Inner chronological split is empty')
    return inner_train, inner_val


def fresh_mlp(hidden: tuple[int, ...], alpha: float, learning_rate: float):
    # partial_fit performs one epoch, allowing validation after each epoch.
    return MLPRegressor(hidden_layer_sizes=hidden, solver='adam', alpha=alpha,
                        learning_rate_init=learning_rate, early_stopping=False,
                        max_iter=1, random_state=42, shuffle=True)


def tune_mlp(train: pd.DataFrame, use_weather: bool, candidates,
             epochs: int, patience: int, val_origins: int, learning_rate: float,
             min_delta: float, issue: pd.Timestamp):
    inner_train, inner_val = inner_split(train, val_origins)
    trans = make_transformer(use_weather)
    xx = trans.fit_transform(inputs(inner_train, use_weather))
    xv = trans.transform(inputs(inner_val, use_weather))
    yt = inner_train.carwashes.to_numpy(dtype=float)
    yv = inner_val.carwashes.to_numpy(dtype=float)
    candidates_results = []
    curves = []
    for hidden, alpha in candidates:
        mlp = fresh_mlp(hidden, alpha, learning_rate)
        best = float('inf')
        best_epoch = 0
        stale = 0
        label = f"{'x'.join(map(str, hidden))}:alpha={alpha:g}"
        for epoch in range(1, epochs + 1):
            mlp.partial_fit(xx, yt)
            train_mae = float(mean_absolute_error(yt, np.maximum(0, mlp.predict(xx))))
            valid_mae = float(mean_absolute_error(yv, np.maximum(0, mlp.predict(xv))))
            curves.append({'issued_at': str(issue), 'feature_set': 'weather' if use_weather else 'calendar',
                           'candidate': label, 'epoch': epoch,
                           'inner_training_mae': train_mae, 'inner_validation_mae': valid_mae})
            if valid_mae < best - min_delta:
                best = valid_mae
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        candidates_results.append({'hidden': hidden, 'alpha': alpha,
                                   'validation_mae': best, 'best_epoch': best_epoch,
                                   'candidate': label})
    winning = min(candidates_results, key=lambda x: x['validation_mae'])
    # Refit on *all outer training examples*, but not on the outer test week.
    # Keep selected number of epochs fixed: the test week never affects stopping.
    full_trans = make_transformer(use_weather)
    full_x = full_trans.fit_transform(inputs(train, use_weather))
    full_y = train.carwashes.to_numpy(dtype=float)
    final = fresh_mlp(winning['hidden'], winning['alpha'], learning_rate)
    for _ in range(winning['best_epoch']):
        final.partial_fit(full_x, full_y)
    return (full_trans, final), winning, curves


def predict_mlp(fitted, test, use_weather):
    trans, net = fitted
    return np.maximum(0, net.predict(trans.transform(inputs(test, use_weather))))


def add_season_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    day = pd.to_datetime(result.local_date, errors='raise').dt.dayofyear
    # Leap-day and year boundary are handled approximately with an annual cycle.
    result['season_sin'] = np.sin(2 * np.pi * (day - 1) / 365.25)
    result['season_cos'] = np.cos(2 * np.pi * (day - 1) / 365.25)
    return result


def add_regimes(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    rain = pd.to_numeric(result.precipitation, errors='coerce')
    past = pd.to_numeric(result.rain_prev72h, errors='coerce')
    # Regimes use archived FORECAST weather, not the weather realized later.
    result['weather_regime'] = np.select(
        [rain >= 0.5, past >= 5.], REGIME_ORDER[:2], default=REGIME_ORDER[2])
    result['forecast_rain_target_mm'] = rain
    result['forecast_rain_prior_72h_mm'] = past
    return result


def grouped_metrics(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=cols+['mae','rmse','wape_percent','observations','signed_bias'])
    chunks=[]
    for name, g in df.groupby(cols, dropna=False, sort=True):
        name = name if isinstance(name, tuple) else (name,)
        entry=dict(zip(cols, name))
        entry.update(core.model_metrics(g.carwashes.to_numpy(),g.prediction.to_numpy()))
        entry['signed_bias']=round(float(np.mean(g.prediction-g.carwashes)),3)
        chunks.append(entry)
    return pd.DataFrame(chunks)


def paired_gains(pred: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    keys = ['issued_at', 'location_id', 'timestamp']
    records=[]
    for algorithm in ('boosted', 'mlp'):
        a = pred[pred.variant == algorithm+'_calendar']
        b = pred[pred.variant == algorithm+'_weather']
        if a.empty or b.empty:
            continue
        cols = keys + ['absolute_error']
        matching = a[cols].merge(b[cols], on=keys, how='outer', suffixes=('_calendar','_weather'),
                                indicator=True, validate='one_to_one')
        if not matching['_merge'].eq('both').all():
            raise RuntimeError(f'{algorithm}: calendar and weather variants evaluated different targets')
        enrich = b[keys+group_cols].drop_duplicates(keys)
        matching=matching.merge(enrich,on=keys,validate='one_to_one')
        groups = matching.groupby(group_cols,dropna=False,sort=True) if group_cols else [((),matching)]
        for vals,g in groups:
            vals = vals if isinstance(vals,tuple) else (vals,)
            entry=dict(zip(group_cols,vals))
            entry['algorithm']=algorithm
            entry['observations']=len(g)
            entry['calendar_mae']=round(float(g.absolute_error_calendar.mean()),3)
            entry['weather_mae']=round(float(g.absolute_error_weather.mean()),3)
            entry['weather_gain_washes_per_hour']=round(float((g.absolute_error_calendar-g.absolute_error_weather).mean()),3)
            records.append(entry)
    return pd.DataFrame(records,columns=group_cols+['algorithm','observations','calendar_mae','weather_mae',
                                                   'weather_gain_washes_per_hour'])


def evaluate(df: pd.DataFrame, window_days: int = 180, test_origins: int = 8,
             algorithms: tuple[str, ...] = ('baseline','boosted','mlp'),
             candidates=None, mlp_epochs: int = 140, mlp_patience: int = 18,
             mlp_validation_origins: int = 2, mlp_lr: float = .001,
             min_delta: float = .01, test_from: str | None = None, test_to: str | None = None,
             min_train_rows: int = 200, min_train_origins: int = 8):
    if candidates is None:
        candidates = parse_candidates('12x6:0.3,24x12:1.5')
    df = add_season_features(df)
    if not algorithms or not set(algorithms).issubset({'baseline','boosted','mlp'}):
        raise ValueError('Only baseline,boosted,mlp are supported')
    if window_days < 1 or test_origins < 1 or mlp_epochs < 1 or mlp_patience < 1 or mlp_validation_origins < 1:
        raise ValueError('Window, origins, epochs and patience must be positive')
    issues=sorted(df.issued_at.unique())
    if test_from:
        issues=[i for i in issues if pd.Timestamp(i).date()>=pd.Timestamp(test_from).date()]
    if test_to:
        issues=[i for i in issues if pd.Timestamp(i).date()<=pd.Timestamp(test_to).date()]
    selected=issues[-test_origins:]
    if not selected:
        raise ValueError('No historical forecast origins matched selected dates')
    all_predictions=[]; curves=[]; mlp_selections=[]; fold_rows=[]
    for issue in selected:
        train,test=backtest.fold_sets(df,issue,window_days)
        if len(train)<min_train_rows or train.issued_at.nunique()<min_train_origins:
            raise ValueError(f'Not enough prior training data for {issue}: {len(train)} rows / '
                             f'{train.issued_at.nunique()} origins. Choose later forecast weeks or a longer window.')
        if (train.timestamp >= issue).any() or (train.issued_at >= issue).any():
            raise RuntimeError('Future label used in training')
        print(f'Backtesting {str(issue)}: {len(train):,} prior rows, {len(test):,} held-out rows',flush=True)
        fold_rows.append({'issued_at':str(issue),'train_rows':len(train),'test_rows':len(test),
                          'earliest_training_issue':str(train.issued_at.min()),
                          'latest_training_target':str(train.timestamp.max())})
        test=add_regimes(test)
        test['target_month']=pd.to_datetime(test.local_date).dt.month
        keep=['issued_at','timestamp','local_date','location_id','hour','dow','lead_day',
              'target_month','carwashes','weather_regime','forecast_rain_target_mm','forecast_rain_prior_72h_mm']
        if 'inferred_zero' in test:
            keep+=['inferred_zero']
        for algorithm in algorithms:
            variants = [False, True] if algorithm != 'baseline' else [False]
            for use_weather in variants:
                variant = algorithm + ('_weather' if use_weather else '_calendar') if algorithm!='baseline' else 'baseline'
                if algorithm=='baseline':
                    fitted=core.Baseline().fit(train)
                    yhat=fitted.predict(test)
                elif algorithm=='boosted':
                    fitted=boost_model(train,use_weather)
                    yhat=fitted.predict(inputs(test,use_weather))
                else:
                    fitted,best,fold_curves=tune_mlp(train,use_weather,candidates,
                        mlp_epochs,mlp_patience,mlp_validation_origins,mlp_lr,min_delta,issue)
                    mlp_selections.append({'issued_at':str(issue),'variant':variant,
                                           'best_candidate':best['candidate'],
                                           'best_inner_mae':round(best['validation_mae'],3),
                                           'best_epoch':best['best_epoch'],
                                           'inner_validation_origins':mlp_validation_origins})
                    curves.extend(fold_curves)
                    yhat=predict_mlp(fitted,test,use_weather)
                yhat=np.maximum(0,yhat)
                out=test[keep].copy()
                out['variant']=variant
                out['prediction']=np.round(yhat,4)
                out['absolute_error']=np.abs(out.carwashes-out.prediction)
                all_predictions.append(out)
    preds=pd.concat(all_predictions,ignore_index=True)
    preds['issue_date']=preds.issued_at.dt.strftime('%Y-%m-%d')
    metrics={
        'overall_metrics.csv':grouped_metrics(preds,['variant']).sort_values('mae'),
        'by_issuance.csv':grouped_metrics(preds,['variant','issue_date']),
        'by_weather_regime.csv':grouped_metrics(preds,['variant','weather_regime']),
        'by_month.csv':grouped_metrics(preds,['variant','target_month']),
        'by_lead_day.csv':grouped_metrics(preds,['variant','lead_day']),
        'by_location.csv':grouped_metrics(preds,['variant','location_id']),
        'by_issue_and_regime.csv':grouped_metrics(preds,['variant','issue_date','weather_regime']),
        'weather_gain_overall.csv':paired_gains(preds,[]),
        'weather_gain_by_issuance.csv':paired_gains(preds,['issue_date']),
        'weather_gain_by_regime.csv':paired_gains(preds,['weather_regime']),
        'weather_gain_by_month.csv':paired_gains(preds,['target_month']),
        'weather_gain_by_lead_day.csv':paired_gains(preds,['lead_day']),
        'mlp_tuning.csv':pd.DataFrame(mlp_selections),
        'mlp_learning_curves.csv':pd.DataFrame(curves),
    }
    if 'inferred_zero' in preds:
        metrics['by_inferred_zero.csv']=grouped_metrics(preds,['variant','inferred_zero'])
    # Each issuance receives equal weight, unlike the all-hours pooled MAE.
    weekly=metrics['by_issuance.csv'].groupby('variant',as_index=False).agg(
        equal_week_mae=('mae','mean'),weeks=('issue_date','nunique'))
    metrics['equal_week_metrics.csv']=weekly.round({'equal_week_mae':3})
    return preds,metrics,fold_rows


def attach_reconstructed_weather(preds: pd.DataFrame, source: str | Path) -> pd.DataFrame:
    # Reconstructed target-time weather is only attached AFTER fitting/predicting.
    # Never feed this into any feature matrix used for model fitting or inference.
    observed = core.load_weather(str(source))
    actual = observed[['location_id','timestamp','precipitation']].rename(
        columns={'precipitation':'reconstructed_target_precip_mm'})
    # Repeated variants are expected; the weather table must have unique site-hour keys.
    if actual.duplicated(['location_id','timestamp']).any():
        raise ValueError('Reconstructed weather contains duplicate site-hour keys')
    result = preds.merge(actual,on=['location_id','timestamp'],how='left',validate='many_to_one')
    amount=result.reconstructed_target_precip_mm
    result['reconstructed_rain_regime']=np.select(
        [amount.isna(), amount>=2., amount>=.5],
        ['Unknown reconstructed rain', 'Heavy reconstructed rain >=2 mm/h',
         'Reconstructed rain 0.5–2 mm/h'], default='Little/no reconstructed rain')
    result['unforecast_rain_event'] = (amount>=2.) & (result.forecast_rain_target_mm<.5)
    return result


def make_parser():
    p=argparse.ArgumentParser(description='Offline weather ablation and MLP chronological training diagnostics')
    p.add_argument('--examples',required=True,help='A saved asof_training_examples.csv, generated with archived forecast weather')
    p.add_argument('--reconstructed-weather-csv',default=None,
                   help='OPTIONAL offline reconstructed target weather for post-prediction diagnostics ONLY; never an input to models')
    p.add_argument('--window-days',type=int,default=180)
    p.add_argument('--test-origins',type=int,default=8)
    p.add_argument('--test-from',default=None,help='First forecast issue date, inclusive (YYYY-MM-DD)')
    p.add_argument('--test-to',default=None,help='Last forecast issue date, inclusive (YYYY-MM-DD)')
    p.add_argument('--models',default='baseline,boosted,mlp')
    p.add_argument('--mlp-candidates',default='12x6:0.3,24x12:1.5')
    p.add_argument('--mlp-epochs',type=int,default=140)
    p.add_argument('--mlp-patience',type=int,default=18)
    p.add_argument('--mlp-validation-origins',type=int,default=2)
    p.add_argument('--mlp-learning-rate',type=float,default=.001)
    p.add_argument('--mlp-min-delta',type=float,default=.01)
    p.add_argument('--output',default='experiments_180d')
    return p


def main(argv=None):
    args=make_parser().parse_args(argv)
    df=backtest.load_examples(args.examples)
    algorithms=tuple(x.strip() for x in args.models.split(',') if x.strip())
    preds,metrics,folds=evaluate(df,window_days=args.window_days,test_origins=args.test_origins,
        algorithms=algorithms,candidates=parse_candidates(args.mlp_candidates),
        mlp_epochs=args.mlp_epochs,mlp_patience=args.mlp_patience,
        mlp_validation_origins=args.mlp_validation_origins,mlp_lr=args.mlp_learning_rate,
        min_delta=args.mlp_min_delta,test_from=args.test_from,test_to=args.test_to)
    if args.reconstructed_weather_csv:
        preds=attach_reconstructed_weather(preds,args.reconstructed_weather_csv)
        metrics['by_reconstructed_rain.csv']=grouped_metrics(preds,['variant','reconstructed_rain_regime'])
        metrics['weather_gain_by_reconstructed_rain.csv']=paired_gains(preds,['reconstructed_rain_regime'])
        metrics['by_unforecast_rain_event.csv']=grouped_metrics(preds,['variant','unforecast_rain_event'])
        metrics['weather_gain_by_unforecast_rain_event.csv']=paired_gains(preds,['unforecast_rain_event'])
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    preds.to_csv(output/'experiment_predictions.csv',index=False)
    for name,table in metrics.items():
        table.to_csv(output/name,index=False)
    meta={'created_at_utc':datetime.now(timezone.utc).isoformat(),
          'examples_source':str(Path(args.examples).resolve()),
          'reconstructed_weather_source':str(Path(args.reconstructed_weather_csv).resolve()) if args.reconstructed_weather_csv else None,
          'window_days':args.window_days,
          'test_origins':len(folds),'models':algorithms,'test_from':args.test_from,'test_to':args.test_to,
          'mlp_candidates':args.mlp_candidates,'mlp_max_epochs':args.mlp_epochs,
          'mlp_patience':args.mlp_patience,'mlp_inner_validation_origins':args.mlp_validation_origins,
          'folds':folds,'important_limits':[
              'This is model-selection validation, not an untouched final test set.',
              'Feature ablation measures predictive contribution, not causal weather effects.',
              'Primary weather regimes use archived forecasts; optional reconstructed weather regimes are POST-HOC only.',
              'Weekly issuance means day-of-week and lead horizon are confounded.',
              'March and October cannot be compared unless evaluation includes both months.',
              'Inferred zeros and site closures require data-quality review.',
              'MLP tuning uses past-only inner validation, then refits for the selected epoch count on outer training labels.'
          ]}
    (output/'experiment_info.json').write_text(json.dumps(meta,indent=2,default=str))
    print('\nPaired experiment results:')
    print(metrics['overall_metrics.csv'].to_string(index=False))
    if len(metrics['weather_gain_overall.csv']):
        print('\nWeather ablation gains (positive = weather helped):')
        print(metrics['weather_gain_overall.csv'].to_string(index=False))
    print(f'\nSaved offline results: {output.resolve()}',flush=True)


if __name__=='__main__':
    main()
