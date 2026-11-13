"""WashCast LightGBM recent-demand feature ablation, preserving as-issued weather.

Add-on: does NOT change production training, labels or previous experiments.
Features were created strictly as-of EACH historical forecast issuance by
recent_demand_features.py.  All variants share identical chronological folds,
weather/storm features, tree hyperparameters, and forecast targets.

This is a development experiment, NOT an untouched statistical test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import backtest
import demand_model_benchmark as original
import experiments
import forecast_reliability as legacy
import model_benchmark as bm
import model_tuning as tuning
import recent_demand_features as rd
import storm_features as storm
import storm_loss_experiment as sl
import washcast as core
import reliability_extended as advanced

# Each set changes ONLY the recent-demand columns. Location identity, weather
# forecasts, storm features, calendar predictors and hyperparameters stay fixed.
ALL_RECENT = tuple(rd.FEATURES)
SHORT_LAGS = ('rd_lag_7d', 'rd_lag_14d', 'rd_lag_28d')
TREND = ('rd_ratio_7d_56d', 'rd_ratio_28d_180d')
STABLE = (
    'rd_daily_mean_28d', 'rd_daily_mean_56d', 'rd_daily_mean_180d',
    'rd_daily_days_28d', 'rd_daily_days_56d', 'rd_daily_days_180d',
    'rd_ratio_28d_180d', 'rd_hour_median_28d',
    'rd_dow_hour_median_56d', 'rd_network_dow_hour_median_56d',
    'rd_store_age_days', 'rd_valid_days_total',
    'rd_days_since_last_valid_day', 'rd_inferred_fraction_28d',
)
SETS = {
    'lgbm_no_recent': (),
    'lgbm_full_recent': ALL_RECENT,
    'lgbm_stable_recent': STABLE,
    'lgbm_no_short_lags': tuple(c for c in ALL_RECENT if c not in SHORT_LAGS),
    'lgbm_no_trend_ratios': tuple(c for c in ALL_RECENT if c not in TREND),
}
REFERENCES = ('weather_median', 'xgb_original')
MODELS = REFERENCES + tuple(SETS)
DEFAULT_MODELS = ','.join(MODELS)
SCRIPT_VERSION = 1


def choose_models(raw: str) -> tuple[str, ...]:
    requested = tuple(v.strip() for v in raw.split(',') if v.strip())
    if (not requested or len(set(requested)) != len(requested)
            or not set(requested).issubset(MODELS)):
        raise ValueError('Choose distinct model names from '+','.join(MODELS))
    if 'weather_median' not in requested or 'lgbm_full_recent' not in requested:
        raise ValueError('Always include weather_median and lgbm_full_recent for paired validation')
    return requested


def input_columns(variant: str) -> tuple[list[str], list[str]]:
    if variant == 'xgb_original':
        return original._columns(False)
    if variant not in SETS:
        raise ValueError(variant)
    return list(core.CAT), (experiments.numeric_cols(True)
                            + list(storm.FEATURES) + list(SETS[variant]))


def make_inputs(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    cat,num = input_columns(variant)
    out = df[cat + num].copy()
    for col in cat:
        out[col] = out[col].astype(str)
    return out


def make_transformer(variant: str) -> ColumnTransformer:
    cat, num = input_columns(variant)
    return ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat),
        ('num', Pipeline([
            ('impute', SimpleImputer(strategy='median',keep_empty_features=True)),
            ('scale', StandardScaler()),
        ]), num),
    ], sparse_threshold=0)


def fold_path(folder: Path, issue: pd.Timestamp, model: str) -> Path:
    return folder/'fold_predictions'/issue.strftime('%Y%m%dT%H%M%S%z')/(model+'.csv')


def _save_csv(path: Path, data: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.partial')
    data.to_csv(temporary,index=False)
    temporary.replace(path)


def run(args):
    models = choose_models(args.models)
    if args.window_days <= 0 or args.validation_origins <= 0 or args.threads <= 0:
        raise ValueError('window-days, validation-origins and threads must be positive')
    required = []
    if any(name.startswith('lgbm_') for name in models):
        required.append('lightgbm')
    if 'xgb_original' in models:
        required.append('xgboost')
    bm.require_libraries(tuple(required))
    frame = original.load_data(args.examples,args.storm_features,args.recent_demand_features)
    issues = tuning.select_validation_issues(frame, args.validation_from,
                                               args.validation_to,args.validation_origins)
    overrides = original._selected_parameters(args.selected_configs)
    folder = Path(args.output)
    manifest = {
        'schema_version': SCRIPT_VERSION,
        'examples': tuning.source_fingerprint(args.examples),
        'storm_features': tuning.source_fingerprint(args.storm_features),
        'recent_demand_features': tuning.source_fingerprint(args.recent_demand_features),
        'selected_configurations': (tuning.source_fingerprint(args.selected_configs)
                                    if args.selected_configs else None),
        'models': models, 'feature_sets': {m:list(SETS[m]) for m in models if m in SETS},
        'window_days':args.window_days, 'validation_from':args.validation_from,
        'validation_to':args.validation_to,'validation_origins':args.validation_origins,
        'issuances':[str(i) for i in issues], 'threads':args.threads,
        'notes':'Development validation: repeated target hours across issuances are correlated; do not claim independent storms.',
    }
    folder.mkdir(parents=True,exist_ok=True)
    mp=folder/'experiment_manifest.json'
    if mp.exists():
        if json.loads(mp.read_text(encoding='utf-8')) != json.loads(json.dumps(manifest)):
            raise ValueError('Output directory used for different experiment; use a fresh --output')
    elif any(folder.glob('fold_predictions/**/*.csv')):
        raise ValueError('Existing predictions without matching manifest; choose new output')
    else:
        mp.write_text(json.dumps(manifest,indent=2),encoding='utf-8')

    for issue in issues:
        outstanding=[m for m in models if not fold_path(folder,issue,m).is_file()]
        if not outstanding:
            print(f'{issue.date()} all models already completed',flush=True)
            continue
        train,test=backtest.fold_sets(frame,issue,args.window_days)
        if len(train)<200 or train.issued_at.nunique()<8 or len(test)==0:
            raise ValueError(f'Insufficient valid training history at {issue}')
        if (train.timestamp>=issue).any() or (train.issued_at>=issue).any():
            raise RuntimeError('Training contains labels/issuances from the future')
        test=experiments.add_regimes(test)
        y=train.carwashes.to_numpy(dtype=float)
        if not np.isfinite(y).all() or (y<0).any():
            raise ValueError('Nonfinite or negative training traffic count')
        fields = ['issued_at','location_id','timestamp','local_date','hour','dow',
                  'lead_day','carwashes','weather_regime','precipitation',
                  'rain_prev72h','rd_store_age_days','rd_daily_days_28d',
                  'rd_daily_mean_28d','rd_daily_mean_180d','rd_ratio_28d_180d']
        if 'inferred_zero' in test:
            fields.append('inferred_zero')
        for model in outstanding:
            if model == 'weather_median':
                predictions=tuning.WeatherAdjustedMedian().fit(train).predict(test)
            else:
                transformer=make_transformer(model)
                xx=transformer.fit_transform(make_inputs(train,model))
                xt=transformer.transform(make_inputs(test,model))
                algorithm='xgboost' if model=='xgb_original' else 'lightgbm'
                estimator=sl.make_estimator(algorithm,'poisson',threads=args.threads,
                                              overrides=overrides.get(algorithm,{}))
                if algorithm=='lightgbm':
                    names=transformer.get_feature_names_out()
                    xx=pd.DataFrame(xx,columns=names)
                    xt=pd.DataFrame(xt,columns=names)
                estimator.fit(xx,y)
                predictions=sl.predict(estimator,algorithm,'poisson',xt)
                del estimator
            pred=np.asarray(predictions,dtype=float).reshape(-1)
            if len(pred)!=len(test) or not np.isfinite(pred).all() or (pred<0).any():
                raise RuntimeError(f'Invalid predictions from {model} at {issue}')
            out=test[fields].copy()
            out['model']=model
            out['prediction']=np.round(pred,6)
            _save_csv(fold_path(folder,issue,model),out)
            print(f'{issue.date()} {model:<24} MAE={np.abs(out.carwashes-pred).mean():.3f} n={len(test)}',flush=True)
    paths=[fold_path(folder,issue,model) for issue in issues for model in models]
    missing=[str(p) for p in paths if not p.is_file()]
    if missing:
        raise RuntimeError(f'{len(missing)} forecast folds incomplete')
    predictions=pd.concat([pd.read_csv(p,dtype={'location_id':str}) for p in paths],ignore_index=True)
    _save_csv(folder/'all_predictions.csv',predictions)
    report_dir=folder/'reliability'
    report_dir.mkdir(exist_ok=True)
    base=legacy.make_reports(predictions,reference='weather_median')
    for name,table in base.items():
        _save_csv(report_dir/name,table)
    extra=advanced.make_reports(predictions, reference='weather_median',
                                comparison='lgbm_no_recent' if 'lgbm_no_recent' in models else 'lgbm_full_recent',
                                min_weather_hours=args.min_weather_hours)
    for name,table in extra.items():
        _save_csv(report_dir/name,table)
    print('\nOverall feature-set evaluation:')
    print(base['overall_metrics.csv'][['model','observations','mae','wape_percent','signed_bias']].to_string(index=False))
    print('\nEqual-store and relative reliability:')
    print(extra['macro_relative_reliability.csv'].to_string(index=False))
    print('Reports:',report_dir.resolve())
    return predictions, base, extra


def make_parser():
    p=argparse.ArgumentParser(description='Compare reduced and full recent-demand feature sets with global LightGBM')
    p.add_argument('--examples',required=True)
    p.add_argument('--storm-features',required=True)
    p.add_argument('--recent-demand-features',required=True)
    p.add_argument('--selected-configs',default=None)
    p.add_argument('--window-days',type=int,default=365)
    p.add_argument('--validation-from',default='2026-05-01')
    p.add_argument('--validation-to',default='2026-07-31')
    p.add_argument('--validation-origins',type=int,default=18)
    p.add_argument('--models',default=DEFAULT_MODELS)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--min-weather-hours',type=int,default=10)
    p.add_argument('--output',default='demand_ablation_may_july')
    return p


if __name__ == '__main__':
    run(make_parser().parse_args())
