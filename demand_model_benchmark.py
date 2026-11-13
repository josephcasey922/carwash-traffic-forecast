"""Offline, resumable WashCast recent-demand forecasting experiment.

PURPOSE: test whether store-specific lag/rolling demand features solve demand
shifts and cold-start problems, without sacrificing weather reliability.

Exact same issued forecast weather, strict chronological training folds, and
identical target rows are used for every model. The existing tuned XGBoost
storm model remains an UNCHANGED reference (with no recent-demand features).

Candidates: XGBoost recent, LightGBM recent, regularized pooled Poisson with
store x weather-regime interactions, existing weather median, adaptive median.

May-July is DEVELOPMENT DATA already examined in earlier experiments. Do not
treat it as an untouched final test. Hold August/September for a frozen test.
No API calls, no company data committed, and no existing code overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import backtest
import experiments
import model_benchmark as bm
import model_tuning as tuning
import storm_features as storm
import storm_loss_experiment as sl
import recent_demand_features as rd
import forecast_reliability as reliability
import washcast as core

KEYS=list(storm.KEYS)
MODELS=('weather_median','adaptive_median','xgb_original','xgb_recent',
        'lightgbm_recent','poisson_recent')
DEFAULT_MODELS=','.join(MODELS)


def parse_models(raw: str) -> tuple[str,...]:
    models=tuple(m.strip().lower() for m in raw.split(',') if m.strip())
    if not models or len(set(models)) != len(models) or not set(models).issubset(MODELS):
        raise ValueError('Choose distinct models from '+','.join(MODELS))
    if 'weather_median' not in models:
        raise ValueError('Include weather_median for paired performance evaluation')
    return models


def load_data(examples_path: str, storm_path: str, recent_path: str) -> pd.DataFrame:
    src=sl.load_enriched(examples_path,storm_path)
    recent=pd.read_csv(recent_path,dtype={'location_id':str})
    needed=set(KEYS+list(rd.FEATURES))
    if not needed.issubset(recent):
        raise ValueError(f'Recent-demand feature CSV missing {sorted(needed-set(recent))}')
    for name in ('issued_at','timestamp'):
        recent[name]=pd.to_datetime(recent[name],utc=True,errors='raise')
    if recent.duplicated(KEYS).any():
        raise ValueError('Duplicate recent-demand feature keys')
    for name in rd.FEATURES:
        recent[name]=pd.to_numeric(recent[name],errors='raise')
    combined=src.merge(recent[KEYS+list(rd.FEATURES)],on=KEYS,
                       how='outer',validate='one_to_one',indicator=True)
    if not combined._merge.eq('both').all() or len(combined)!=len(src):
        raise ValueError('Recent feature file must match EVERY (issuance,store,target) example exactly')
    return combined.drop(columns='_merge')


def _columns(recent: bool) -> tuple[list[str],list[str]]:
    cat=list(core.CAT)
    num=experiments.numeric_cols(True)+list(storm.FEATURES)
    if recent:
        num+=list(rd.FEATURES)
    return cat,num


def _transformer(recent: bool, *, linear: bool=False):
    cat, num=_columns(recent)
    if linear:
        cat=cat+['store_weather_regime']
    return ColumnTransformer([
        ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=linear),cat),
        ('num',Pipeline([
            ('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
            ('scale',StandardScaler())]),num)
    ],sparse_threshold=.3 if linear else 0)


def _inputs(df: pd.DataFrame, recent: bool, *, linear: bool=False) -> pd.DataFrame:
    cat,num=_columns(recent)
    x=df[cat+num].copy()
    for key in cat:
        x[key]=x[key].astype(str)
    if linear:
        # Coefficients shrink toward a global weather response under L2.
        # This is a fixed-effects interaction model, not a Bayesian mixed model.
        regimes=experiments.add_regimes(df).weather_regime.astype(str)
        x['store_weather_regime']=df.location_id.astype(str)+'|'+regimes
    return x


def _selected_parameters(path: str | None) -> dict:
    if not path:
        return {}
    overrides,_=sl.tuned_overrides(path)
    return overrides


def _adaptive_median(weather_values: np.ndarray, test: pd.DataFrame) -> np.ndarray:
    """Fixed, support-shrunk 28/180-day ratio correction; not fitted on test labels."""
    r=pd.to_numeric(test.rd_ratio_28d_180d,errors='coerce').to_numpy(dtype=float)
    count=pd.to_numeric(test.rd_daily_days_28d,errors='coerce').fillna(0).to_numpy(dtype=float)
    # Several recent complete days are needed; cold-start stays near pooled
    # weather baseline. Fixed conservative cap prevents runaway corrections.
    shrink=count/(count+14.)
    r=np.where(np.isfinite(r),r,1.)
    factor=1.+shrink*(np.clip(r,.55,1.45)-1.)
    return np.maximum(0.,weather_values*factor)


def prediction_path(out: Path, issue: pd.Timestamp, name: str) -> Path:
    return out/'fold_predictions'/issue.strftime('%Y%m%dT%H%M%S%z')/(name+'.csv')


def save_atomic(path: Path, part: pd.DataFrame):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.csv.partial')
    part.to_csv(tmp,index=False)
    tmp.replace(path)


def _manifest(args, models: tuple[str,...], issues: list[pd.Timestamp]) -> dict:
    return {
        'schema_version':1,
        'examples':tuning.source_fingerprint(args.examples),
        'storm_features':tuning.source_fingerprint(args.storm_features),
        'recent_demand_features':tuning.source_fingerprint(args.recent_demand_features),
        'selected_configs':tuning.source_fingerprint(args.selected_configs) if args.selected_configs else None,
        'window_days':args.window_days,
        'validation_from':args.validation_from,
        'validation_to':args.validation_to,
        'validation_origins':args.validation_origins,
        'issuances':[str(issue) for issue in issues],
        'models':models,'threads':args.threads,'linear_alpha':args.linear_alpha,
        'notes':[
            'Development backtest, not an untouched test.',
            'All target labels must precede the outer forecast issuance; feature rows are strictly as-of their own issuance.',
            'Historical repeated two-day target labels deduplicated before constructing lags and rolling statistics.',
            'The xgb_original variant uses original storm features and exactly the selected XGBoost parameters.',
            'The weather_median remains a separate consistent reference, not a human forecast.',
            'Forecast rain is not necessarily realized rain; actual storm events require observed weather.',
            'Site-days with missing/inferred hours may represent closures or measurement gaps.',
            'Per-site store ages measure observed valid traffic history; they are not verified opening dates.',
        ],
    }


def _prepare_output(folder: Path, spec: dict):
    folder.mkdir(parents=True,exist_ok=True)
    p=folder/'experiment_manifest.json'
    if p.exists():
        if json.loads(p.read_text(encoding='utf-8')) != json.loads(json.dumps(spec)):
            raise ValueError('This output directory belongs to a different experiment; use a new --output')
    elif list(folder.glob('fold_predictions/**/*.csv')):
        raise ValueError('Existing model predictions without matching manifest')
    else:
        p.write_text(json.dumps(spec,indent=2),encoding='utf-8')


def run(args):
    models=parse_models(args.models)
    if args.window_days<1 or args.threads<1 or args.linear_alpha<0 or args.validation_origins<1:
        raise ValueError('Positive window, threads, origin count and nonnegative linear_alpha required')
    libs=('xgboost',) if any(m.startswith('xgb_') for m in models) else ()
    if 'lightgbm_recent' in models:
        libs+=('lightgbm',)
    bm.require_libraries(libs)
    df=load_data(args.examples,args.storm_features,args.recent_demand_features)
    issues=tuning.select_validation_issues(df,args.validation_from,
                                           args.validation_to,args.validation_origins)
    overrides=_selected_parameters(args.selected_configs)
    folder=Path(args.output)
    _prepare_output(folder,_manifest(args,models,issues))
    for issue in issues:
        outstanding=[m for m in models if not prediction_path(folder,issue,m).exists()]
        if not outstanding:
            print(f'Resuming {issue}: already complete',flush=True)
            continue
        train,test=backtest.fold_sets(df,issue,args.window_days)
        if len(train)<200 or train.issued_at.nunique()<8 or test.empty:
            raise ValueError(f'Insufficient historical labeled training data for {issue}')
        if (train.timestamp>=issue).any() or (train.issued_at>=issue).any():
            raise RuntimeError('A future target leaked into the training fold')
        test=experiments.add_regimes(test)
        y=train.carwashes.to_numpy(dtype=float)
        if not np.isfinite(y).all() or (y<0).any():
            raise ValueError('Invalid training counts')
        fields=KEYS+['local_date','hour','dow','lead_day','carwashes',
                     'weather_regime','precipitation','rain_prev72h',
                     'rd_store_age_days','rd_daily_days_28d','rd_daily_mean_28d',
                     'rd_daily_mean_180d','rd_ratio_28d_180d']
        if 'inferred_zero' in test:
            fields.append('inferred_zero')
        median_fitted=None
        median_prediction=None
        if 'weather_median' in outstanding or 'adaptive_median' in outstanding:
            median_fitted=tuning.WeatherAdjustedMedian().fit(train)
            median_prediction=median_fitted.predict(test)
        # Use one transform per feature set: all tree variants with new features
        # receive the EXACT same matrix and target rows.
        transformed={}
        for label in ('original','recent','linear'):
            needed=(label=='original' and 'xgb_original' in outstanding) or (label=='recent' and any(m in outstanding for m in ('xgb_recent','lightgbm_recent'))) or (label=='linear' and 'poisson_recent' in outstanding)
            if not needed:continue
            new=label != 'original'
            linear=label=='linear'
            proc=_transformer(new,linear=linear)
            xx=proc.fit_transform(_inputs(train,new,linear=linear))
            xt=proc.transform(_inputs(test,new,linear=linear))
            transformed[label]=(xx,xt,proc.get_feature_names_out())
        for model in outstanding:
            if model=='weather_median':
                forecast=median_prediction
            elif model=='adaptive_median':
                forecast=_adaptive_median(median_prediction,test)
            elif model=='poisson_recent':
                xx,xt,_=transformed['linear']
                estimator=PoissonRegressor(alpha=args.linear_alpha,fit_intercept=True,
                                           max_iter=180,tol=1e-4)
                estimator.fit(xx,y)
                forecast=estimator.predict(xt)
            else:
                if model=='xgb_original':
                    family='xgboost';label='original'
                elif model=='xgb_recent':
                    family='xgboost';label='recent'
                elif model=='lightgbm_recent':
                    family='lightgbm';label='recent'
                else:
                    raise ValueError(model)
                xx,xt,names=transformed[label]
                estimator=sl.make_estimator(family,'poisson',threads=args.threads,
                                         overrides=overrides.get(family,{}))
                if family=='lightgbm':
                    xx=pd.DataFrame(xx,columns=names)
                    xt=pd.DataFrame(xt,columns=names)
                estimator.fit(xx,y)
                forecast=sl.predict(estimator,family,'poisson',xt)
                del estimator
            forecast=np.asarray(forecast,dtype=float).reshape(-1)
            if len(forecast)!=len(test) or not np.isfinite(forecast).all() or (forecast<0).any():
                raise RuntimeError(f'Invalid predictions from {model} at {issue}')
            result=test[fields].copy()
            result['model']=model
            result['prediction']=np.round(forecast,6)
            save_atomic(prediction_path(folder,issue,model),result)
            mae=float(np.mean(np.abs(result.carwashes-result.prediction)))
            print(f'{issue.date()} {model:<24} MAE={mae:.3f} n={len(test)}',flush=True)
    paths=[prediction_path(folder,i,m) for i in issues for m in models]
    missing=[str(p) for p in paths if not p.exists()]
    if missing:
        raise RuntimeError(f'Experiment incomplete: {len(missing)} fold files missing')
    preds=pd.concat([pd.read_csv(p,dtype={'location_id':str}) for p in paths],ignore_index=True)
    reports=reliability.make_reports(preds,reference='weather_median')
    preds.to_csv(folder/'all_predictions.csv',index=False)
    out=folder/'reliability'
    out.mkdir(exist_ok=True)
    for name,frame in reports.items():
        frame.to_csv(out/name,index=False)
    print('\nGlobal recent-demand comparison (development validation):')
    print(reports['overall_metrics.csv'].to_string(index=False),flush=True)
    print('\nEqual-store macro evaluation:')
    print(reports['store_macro_metrics.csv'].to_string(index=False),flush=True)
    print('\nReports:',out.resolve(),flush=True)
    return preds,reports


def make_parser():
    p=argparse.ArgumentParser(description='Offline multi-store autoregressive forecast benchmark')
    p.add_argument('--examples',required=True)
    p.add_argument('--storm-features',required=True)
    p.add_argument('--recent-demand-features',required=True)
    p.add_argument('--selected-configs',default=None)
    p.add_argument('--window-days',type=int,default=365)
    p.add_argument('--validation-from',default='2026-05-01')
    p.add_argument('--validation-to',default='2026-07-31')
    p.add_argument('--validation-origins',type=int,default=12)
    p.add_argument('--models',default=DEFAULT_MODELS)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--linear-alpha',type=float,default=0.15)
    p.add_argument('--output',default='demand_benchmark_may_july')
    return p


def main(argv=None):
    return run(make_parser().parse_args(argv))

if __name__=='__main__':
    main()
