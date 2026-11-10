"""Global-model challenger benchmark on WashCast's saved as-of examples.

The models receive identical inputs and are evaluated on identical held-out
issuance/site/target-hour rows. Weather inputs come ONLY from the already-built
as-issued archived-forecast training examples. No weather API calls are made.

All outer training labels have target timestamps strictly before the issuance
being predicted. Fixed starter hyperparameters are used; do not interpret this
as a hyperparameter-tuned or final untouched model comparison.
"""
from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import backtest
import experiments
import washcast as core

ALL_MODELS = ('baseline', 'boosted', 'lightgbm', 'xgboost', 'catboost')
KEYS = ['issued_at', 'location_id', 'timestamp']


def parse_models(value: str) -> tuple[str, ...]:
    result = tuple(s.strip().lower() for s in value.split(',') if s.strip())
    if not result or len(set(result)) != len(result) or not set(result).issubset(ALL_MODELS):
        raise ValueError(f'Choose distinct model names from: {", ".join(ALL_MODELS)}')
    # A paired benchmark against boosted requires boosted in the selection.
    if 'boosted' not in result:
        raise ValueError('Include boosted in --models to retain the established reference')
    return result


def require_libraries(models: tuple[str, ...]) -> dict[str, str]:
    names = {'lightgbm': 'lightgbm', 'xgboost': 'xgboost', 'catboost': 'catboost'}
    versions = {}
    missing = []
    for model in models:
        package = names.get(model)
        if not package:
            continue
        try:
            mod = importlib.import_module(package)
            versions[package] = str(getattr(mod, '__version__', 'unknown'))
        except ImportError:
            missing.append(package)
    if missing:
        raise RuntimeError('Required optional packages are not installed: ' + ', '.join(missing) +
                           '. Run: .\\.venv\\Scripts\\python.exe -m pip install ' + ' '.join(missing))
    return versions


def make_estimator(name: str, threads: int = 2):
    """Count-appropriate starter configurations, fixed before examining test labels."""
    if name == 'boosted':
        return HistGradientBoostingRegressor(
            loss='poisson', max_iter=130, max_leaf_nodes=19,
            learning_rate=.07, l2_regularization=.5, random_state=42)
    if name == 'lightgbm':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            objective='poisson', n_estimators=220, learning_rate=.06,
            num_leaves=19, min_child_samples=35, reg_lambda=1.0,
            verbosity=-1, n_jobs=threads, random_state=42)
    if name == 'xgboost':
        from xgboost import XGBRegressor
        return XGBRegressor(
            objective='count:poisson', n_estimators=230, learning_rate=.06,
            max_depth=5, min_child_weight=5, subsample=.9,
            colsample_bytree=.9, reg_lambda=1.0, tree_method='hist',
            n_jobs=threads, random_state=42, verbosity=0)
    if name == 'catboost':
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            loss_function='Poisson', iterations=180, depth=6,
            learning_rate=.06, l2_leaf_reg=3.0, random_seed=42,
            thread_count=threads, verbose=False, allow_writing_files=False)
    raise ValueError(f'Unknown estimator {name!r}')


def response_prediction(estimator, name: str, X: np.ndarray) -> np.ndarray:
    # For CatBoost Poisson, explicitly request predicted counts, not log rates.
    if name == 'catboost':
        pred = estimator.predict(X, prediction_type='Exponent')
    else:
        pred = estimator.predict(X)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    if len(pred) != len(X) or not np.isfinite(pred).all():
        raise RuntimeError(f'{name} returned invalid predictions')
    return np.maximum(0., pred)


def select_issuances(df: pd.DataFrame, test_origins: int,
                     test_from: str | None, test_to: str | None) -> list[pd.Timestamp]:
    issues = sorted(df.issued_at.unique())
    if test_from:
        date = pd.Timestamp(test_from).date()
        issues = [i for i in issues if pd.Timestamp(i).date() >= date]
    if test_to:
        date = pd.Timestamp(test_to).date()
        issues = [i for i in issues if pd.Timestamp(i).date() <= date]
    if test_origins < 1 or not issues:
        raise ValueError('No forecast issuances selected, or --test-origins is not positive')
    return issues[-test_origins:]


def paired_improvements(preds: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """MAE(boosted) - MAE(challenger) on the same labeled targets."""
    ref = preds.loc[preds.model.eq('boosted'), KEYS + ['carwashes', 'absolute_error']]
    if ref.empty or ref.duplicated(KEYS).any():
        raise RuntimeError('Missing/duplicate reference predictions')
    records = []
    for challenger in sorted(preds.model.unique()):
        if challenger == 'boosted':
            continue
        other = preds.loc[preds.model.eq(challenger),
                          KEYS + [c for c in groups if c not in KEYS] + ['carwashes', 'absolute_error']]
        if other.duplicated(KEYS).any():
            raise RuntimeError(f'Duplicate test target keys for {challenger}')
        aligned = ref.merge(other, on=KEYS, how='outer', validate='one_to_one',
                            suffixes=('_boosted', '_challenger'), indicator=True)
        if not aligned._merge.eq('both').all():
            raise RuntimeError(f'{challenger} was evaluated on different targets')
        if not np.allclose(aligned.carwashes_boosted, aligned.carwashes_challenger):
            raise RuntimeError(f'{challenger}: actual target values differ')
        partitions = aligned.groupby(groups, dropna=False, sort=True) if groups else [((), aligned)]
        for label, chunk in partitions:
            label = label if isinstance(label, tuple) else (label,)
            item = dict(zip(groups, label))
            item.update({
                'challenger': challenger,
                'observations': len(chunk),
                'boosted_mae': round(float(chunk.absolute_error_boosted.mean()), 3),
                'challenger_mae': round(float(chunk.absolute_error_challenger.mean()), 3),
                'improvement_vs_boosted': round(float(
                    (chunk.absolute_error_boosted - chunk.absolute_error_challenger).mean()), 3),
            })
            records.append(item)
    return pd.DataFrame(records, columns=groups + [
        'challenger','observations','boosted_mae','challenger_mae','improvement_vs_boosted'])


def evaluate(df: pd.DataFrame, *, window_days: int = 365, test_origins: int = 7,
             test_from: str | None = None, test_to: str | None = None,
             models: tuple[str, ...] = ALL_MODELS, threads: int = 2,
             min_train_rows: int = 200, min_train_origins: int = 8):
    if window_days < 1 or threads < 1:
        raise ValueError('Window and thread counts must be positive')
    parse_models(','.join(models))
    require_libraries(models)
    df = experiments.add_season_features(df)
    selected = select_issuances(df, test_origins, test_from, test_to)
    predictions, folds = [], []
    for issue in selected:
        train, test = backtest.fold_sets(df, issue, window_days)
        if len(train) < min_train_rows or train.issued_at.nunique() < min_train_origins:
            raise ValueError(f'Insufficient pre-issuance training data at {issue}: '
                             f'{len(train)} rows, {train.issued_at.nunique()} issuances')
        if test.empty or (train.timestamp >= issue).any() or (train.issued_at >= issue).any():
            raise RuntimeError('Missing held-out rows or future targets in training')
        print(f'Benchmark {issue}: {len(train):,} train rows, '
              f'{len(test):,} test rows, models={",".join(models)}', flush=True)
        folds.append({'issued_at': str(issue), 'train_rows': len(train),
                      'test_rows': len(test), 'latest_training_target': str(train.timestamp.max())})

        test = experiments.add_regimes(test)
        test['target_month'] = pd.to_datetime(test.local_date).dt.month
        fields = KEYS + ['local_date','hour','dow','lead_day','target_month',
                         'carwashes','weather_regime','forecast_rain_target_mm',
                         'forecast_rain_prior_72h_mm']
        if 'inferred_zero' in test.columns:
            fields.append('inferred_zero')
        y = train.carwashes.to_numpy(dtype=float)
        if not np.isfinite(y).all() or np.any(y < 0):
            raise ValueError('Training counts must be finite and nonnegative')
        # Fit ONE shared transformation for all challenger algorithms in a fold.
        # This is the same transformation used by experiments.boost_model.
        need_transformed = any(m != 'baseline' for m in models)
        if need_transformed:
            transform = experiments.make_transformer(True)
            X_train = transform.fit_transform(experiments.inputs(train, True))
            X_test = transform.transform(experiments.inputs(test, True))
        # LGBM's sklearn wrapper prefers named features for BOTH fit and predict.
        # This retains the exact same numeric matrix while avoiding a feature-name warning.
        if 'lightgbm' in models:
            feature_names = transform.get_feature_names_out()
            X_train_lgbm = pd.DataFrame(X_train, columns=feature_names)
            X_test_lgbm = pd.DataFrame(X_test, columns=feature_names)
        for name in models:
            if name == 'baseline':
                pred = np.maximum(0., core.Baseline().fit(train).predict(test))
            else:
                estimator = make_estimator(name, threads)
                fit_x = X_train_lgbm if name == 'lightgbm' else X_train
                test_x = X_test_lgbm if name == 'lightgbm' else X_test
                estimator.fit(fit_x, y)
                pred = response_prediction(estimator, name, test_x)
            out = test[fields].copy()
            out['model'] = name
            out['prediction'] = np.round(pred, 4)
            out['absolute_error'] = np.abs(out.carwashes - out.prediction)
            predictions.append(out)
    preds = pd.concat(predictions, ignore_index=True)
    preds['issue_date'] = preds.issued_at.dt.strftime('%Y-%m-%d')
    reports = {
        'overall_metrics.csv': experiments.grouped_metrics(preds, ['model']).sort_values('mae'),
        'by_issuance.csv': experiments.grouped_metrics(preds, ['model','issue_date']),
        'by_weather_regime.csv': experiments.grouped_metrics(preds, ['model','weather_regime']),
        'by_location.csv': experiments.grouped_metrics(preds, ['model','location_id']),
        'by_lead_day.csv': experiments.grouped_metrics(preds, ['model','lead_day']),
        'by_issue_and_regime.csv': experiments.grouped_metrics(preds, ['model','issue_date','weather_regime']),
        'gain_vs_boosted.csv': paired_improvements(preds, []),
        'gain_vs_boosted_by_regime.csv': paired_improvements(preds, ['weather_regime']),
        'gain_vs_boosted_by_issuance.csv': paired_improvements(preds, ['issue_date']),
        'gain_vs_boosted_by_location.csv': paired_improvements(preds, ['location_id']),
    }
    if 'inferred_zero' in preds:
        reports['by_inferred_zero.csv'] = experiments.grouped_metrics(preds, ['model','inferred_zero'])
    return preds, reports, folds


def make_parser():
    parser = argparse.ArgumentParser(description='Offline, leakage-safe global WashCast model benchmark')
    parser.add_argument('--examples', required=True)
    parser.add_argument('--window-days', type=int, default=365)
    parser.add_argument('--test-origins', type=int, default=7)
    parser.add_argument('--test-from', default=None)
    parser.add_argument('--test-to', default=None)
    parser.add_argument('--models', default=','.join(ALL_MODELS))
    parser.add_argument('--threads', type=int, default=2,
                        help='CPU threads per third-party model (default 2; suitable for 8 GB RAM)')
    parser.add_argument('--reconstructed-weather-csv', default=None,
                        help='Optional observed/reconstructed weather for post-prediction diagnostics ONLY')
    parser.add_argument('--output', default='model_benchmark_365d')
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    selected = parse_models(args.models)
    installed = require_libraries(selected)
    df = backtest.load_examples(args.examples)
    preds, reports, folds = evaluate(
        df, window_days=args.window_days, test_origins=args.test_origins,
        test_from=args.test_from, test_to=args.test_to,
        models=selected, threads=args.threads)
    if args.reconstructed_weather_csv:
        preds = experiments.attach_reconstructed_weather(preds, args.reconstructed_weather_csv)
        reports['by_reconstructed_rain.csv'] = experiments.grouped_metrics(
            preds, ['model','reconstructed_rain_regime'])
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    preds.to_csv(folder/'benchmark_predictions.csv', index=False)
    for name, report in reports.items():
        report.to_csv(folder/name, index=False)
    meta = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'examples_source': str(Path(args.examples).resolve()),
        'models': selected, 'third_party_versions': installed,
        'scikit_learn_version': importlib.import_module('sklearn').__version__,
        'window_days': args.window_days, 'test_from': args.test_from,
        'test_to': args.test_to, 'test_origins': len(folds),
        'threads': args.threads, 'folds': folds,
        'estimator_params': {
            name: make_estimator(name, args.threads).get_params()
            for name in selected if name != 'baseline'},
        'notes': [
            'Model-selection validation; these are NOT untouched final test data.',
            'All models share identical as-of features, preprocessing and held-out targets.',
            'Each training label timestamp strictly precedes the issue being predicted.',
            'Fixed starter hyperparameters; more rigorous tuning requires separate past-only validation.',
            'CatBoost intentionally uses the SAME one-hot inputs as other models, not native categorical handling.',
            'Weather regimes use archived forecast weather; reconstructed rain is post-prediction diagnostic only.',
            'Forecast issuances overlap in target hours; observations are not independent.',
        ],
    }
    (folder/'benchmark_info.json').write_text(json.dumps(meta, indent=2, default=str))
    print('\nGlobal benchmark performance (lower MAE is better):')
    print(reports['overall_metrics.csv'].to_string(index=False))
    print('\nPaired improvements relative to the existing boosted model (positive=challenger better):')
    print(reports['gain_vs_boosted.csv'].to_string(index=False))
    print(f'\nSaved offline benchmark results to: {folder.resolve()}')


if __name__ == '__main__':
    main()
