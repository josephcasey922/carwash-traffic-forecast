"""Offline weather feature-group ablations for WashCast's *global* boosted model.

Uses exactly the same chronological train/test folds, gradient-boosting settings,
preprocessing, and as-issued weather features as experiments.py. Does not make API
requests, retrain the production model, or alter input data.

Two complementary tests per weather group:
  (1) calendar + just this group (versus calendar only), and
  (2) full weather minus this group (versus full weather).
The latter shows incremental value conditional on all other weather variables.
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import backtest
import experiments
import washcast as core

WEATHER_GROUPS = {
    'temperature': ('temperature_2m',),
    'target_rain': ('precipitation',),
    'prior_rain': ('rain_prev24h', 'rain_prev72h'),
    'clouds': ('cloud_cover',),
}
KEYS = ['issued_at', 'location_id', 'timestamp']


def feature_sets() -> dict[str, tuple[str, ...]]:
    full = tuple(experiments.WEATHER_NUM)
    if set().union(*(set(names) for names in WEATHER_GROUPS.values())) != set(full):
        raise RuntimeError('Weather groups must partition the current weather feature columns')
    result = {'calendar_only': (), 'full_weather': full}
    for group, cols in WEATHER_GROUPS.items():
        result[f'{group}_only'] = tuple(cols)
        result[f'without_{group}'] = tuple(c for c in full if c not in cols)
    return result


def numeric_features(weather: tuple[str, ...]) -> list[str]:
    """Retain original core.NUM ordering to reproduce existing experiment fits."""
    enabled = set(weather)
    if not enabled <= set(experiments.WEATHER_NUM):
        raise ValueError(f'Unknown weather columns: {sorted(enabled-set(experiments.WEATHER_NUM))}')
    return ([c for c in core.NUM if c not in experiments.WEATHER_NUM or c in enabled]
            + list(experiments.SEASON_NUM))


def make_boosted_model(weather: tuple[str, ...]):
    numeric = numeric_features(weather)
    transform = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), list(core.CAT)),
        ('num', Pipeline([('fill', SimpleImputer(strategy='median')),
                          ('scale', StandardScaler())]), numeric),
    ], sparse_threshold=0)
    return Pipeline([
        ('features', transform),
        ('model', HistGradientBoostingRegressor(
            loss='poisson', max_iter=130, max_leaf_nodes=19,
            learning_rate=.07, l2_regularization=.5, random_state=42)),
    ])


def model_inputs(frame: pd.DataFrame, weather: tuple[str, ...]) -> pd.DataFrame:
    cols = list(core.CAT) + numeric_features(weather)
    result = frame[cols].copy()
    for col in core.CAT:
        result[col] = result[col].astype(str)
    return result


def paired_gain(preds: pd.DataFrame, reference: str, challenger: str,
                group_cols: list[str] | None = None) -> pd.DataFrame:
    """Positive improvement means challenger MAE is lower on identical targets."""
    group_cols = group_cols or []
    left = preds.loc[preds.feature_set.eq(reference),
                     KEYS + group_cols + ['carwashes', 'absolute_error']]
    right = preds.loc[preds.feature_set.eq(challenger),
                      KEYS + ['carwashes', 'absolute_error']]
    if left.empty or right.empty:
        raise ValueError(f'Missing feature set in paired comparison: {reference}, {challenger}')
    if left.duplicated(KEYS).any() or right.duplicated(KEYS).any():
        raise ValueError('Duplicate issued-at / site / target-hour combinations')
    merged = left.merge(right, on=KEYS, how='outer', validate='one_to_one',
                        suffixes=('_ref', '_challenger'), indicator=True)
    if not merged['_merge'].eq('both').all():
        raise RuntimeError('Feature sets have different evaluation target rows')
    if not np.allclose(merged.carwashes_ref, merged.carwashes_challenger):
        raise RuntimeError('Feature sets disagree on actual traffic values')
    pieces = merged.groupby(group_cols, dropna=False, sort=True) if group_cols else [((), merged)]
    rows = []
    for value, group in pieces:
        value = value if isinstance(value, tuple) else (value,)
        record = dict(zip(group_cols, value))
        record.update({
            'observations': len(group),
            'reference_mae': round(float(group.absolute_error_ref.mean()), 3),
            'challenger_mae': round(float(group.absolute_error_challenger.mean()), 3),
            'gain_washes_per_hour': round(float((group.absolute_error_ref - group.absolute_error_challenger).mean()), 3),
        })
        rows.append(record)
    return pd.DataFrame(rows, columns=group_cols + ['observations','reference_mae',
                                                     'challenger_mae','gain_washes_per_hour'])


def group_gain_reports(preds: pd.DataFrame, regime_col: str | None = None) -> dict[str, pd.DataFrame]:
    groups = [regime_col] if regime_col else []
    removed = []
    added = []
    for group in WEATHER_GROUPS:
        minus = paired_gain(preds, f'without_{group}', 'full_weather', groups)
        minus.insert(len(groups), 'feature_group', group)
        removed.append(minus)
        alone = paired_gain(preds, 'calendar_only', f'{group}_only', groups)
        alone.insert(len(groups), 'feature_group', group)
        added.append(alone)
    suffix = f'_by_{regime_col}' if regime_col else ''
    return {
        f'leave_one_out_gain{suffix}.csv': pd.concat(removed, ignore_index=True),
        f'single_group_gain{suffix}.csv': pd.concat(added, ignore_index=True),
    }


def evaluate(df: pd.DataFrame, window_days: int = 365, test_origins: int = 28,
             test_from: str | None = None, test_to: str | None = None,
             min_train_rows: int = 200, min_train_origins: int = 8):
    if window_days < 1 or test_origins < 1:
        raise ValueError('Window days and test origins must be positive')
    df = experiments.add_season_features(df)
    issues = sorted(df.issued_at.unique())
    if test_from:
        issues = [i for i in issues if pd.Timestamp(i).date() >= pd.Timestamp(test_from).date()]
    if test_to:
        issues = [i for i in issues if pd.Timestamp(i).date() <= pd.Timestamp(test_to).date()]
    selected = issues[-test_origins:]
    if not selected:
        raise ValueError('No forecast issuances matched the selected dates')

    configurations = feature_sets()
    predictions = []
    fold_info = []
    for issue in selected:
        train, test = backtest.fold_sets(df, issue, window_days)
        if len(train) < min_train_rows or train.issued_at.nunique() < min_train_origins:
            raise ValueError(f'Insufficient pre-issuance training data at {issue}: '
                             f'{len(train)} examples and {train.issued_at.nunique()} issuances')
        if train.empty or test.empty or (train.timestamp >= issue).any() or (train.issued_at >= issue).any():
            raise RuntimeError('Missing test rows or future labels in training data')
        test = experiments.add_regimes(test)
        test['target_month'] = pd.to_datetime(test.local_date).dt.month
        columns = KEYS + ['local_date','hour','dow','lead_day','target_month','carwashes',
                          'weather_regime','forecast_rain_target_mm','forecast_rain_prior_72h_mm']
        if 'inferred_zero' in test:
            columns.append('inferred_zero')
        print(f'Feature ablation {issue}: {len(train):,} prior rows, '
              f'{len(test):,} test rows, {len(configurations)} models', flush=True)
        fold_info.append({'issued_at': str(issue), 'train_rows': len(train),
                          'test_rows': len(test), 'latest_training_target': str(train.timestamp.max())})
        target = train.carwashes.to_numpy(dtype=float)
        for label, fields in configurations.items():
            fitted = make_boosted_model(fields)
            fitted.fit(model_inputs(train, fields), target)
            estimate = np.maximum(0, fitted.predict(model_inputs(test, fields)))
            result = test[columns].copy()
            result['feature_set'] = label
            result['prediction'] = np.round(estimate, 4)
            result['absolute_error'] = abs(result.carwashes - result.prediction)
            predictions.append(result)

    all_preds = pd.concat(predictions, ignore_index=True)
    all_preds['issue_date'] = all_preds.issued_at.dt.strftime('%Y-%m-%d')
    reports = {
        'ablation_overall.csv': experiments.grouped_metrics(all_preds, ['feature_set']).sort_values('mae'),
        'ablation_by_weather_regime.csv': experiments.grouped_metrics(all_preds, ['feature_set','weather_regime']),
        'ablation_by_location.csv': experiments.grouped_metrics(all_preds, ['feature_set','location_id']),
        'ablation_by_issuance.csv': experiments.grouped_metrics(all_preds, ['feature_set','issue_date']),
        'ablation_by_lead_day.csv': experiments.grouped_metrics(all_preds, ['feature_set','lead_day']),
    }
    if 'inferred_zero' in all_preds:
        reports['ablation_by_inferred_zero.csv'] = experiments.grouped_metrics(all_preds,['feature_set','inferred_zero'])
    reports.update(group_gain_reports(all_preds))
    reports.update(group_gain_reports(all_preds, 'weather_regime'))
    return all_preds, reports, fold_info


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Leakage-safe offline global-model weather feature ablation')
    parser.add_argument('--examples', required=True)
    parser.add_argument('--window-days', type=int, default=365)
    parser.add_argument('--test-origins', type=int, default=28)
    parser.add_argument('--test-from', default=None)
    parser.add_argument('--test-to', default=None)
    parser.add_argument('--reconstructed-weather-csv', default=None,
                        help='Optional post-prediction diagnostics only; NOT a model feature')
    parser.add_argument('--output', default='feature_ablation_365d')
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    data = backtest.load_examples(args.examples)
    predictions, reports, folds = evaluate(data, args.window_days, args.test_origins,
                                            args.test_from, args.test_to)
    if args.reconstructed_weather_csv:
        predictions = experiments.attach_reconstructed_weather(predictions, args.reconstructed_weather_csv)
        reports['ablation_by_reconstructed_rain.csv'] = experiments.grouped_metrics(
            predictions, ['feature_set', 'reconstructed_rain_regime'])
        reports.update(group_gain_reports(predictions, 'reconstructed_rain_regime'))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / 'feature_ablation_predictions.csv', index=False)
    for filename, data_frame in reports.items():
        data_frame.to_csv(output / filename, index=False)
    info = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'examples_source': str(Path(args.examples).resolve()),
        'window_days': args.window_days, 'test_from': args.test_from, 'test_to': args.test_to,
        'test_origins': len(folds), 'feature_sets': feature_sets(), 'folds': folds,
        'constraints': [
            'All models use identical held-out issued-at / site / target-hour rows.',
            'All labels in each training fold precede the issued-at time.',
            'Input weather reflects as-of data in the existing training examples.',
            'Group contributions are predictive, not causal; feature groups are correlated.',
            'These results are model-selection validation, not an untouched final test.',
            'Reconstructed target rainfall is post-prediction diagnostic only.',
        ],
    }
    (output / 'feature_ablation_info.json').write_text(json.dumps(info, indent=2, default=str))
    print('\nFeature-set performance (lower MAE is better):')
    print(reports['ablation_overall.csv'][['feature_set','mae','observations','signed_bias']].to_string(index=False))
    print('\nLeave-one-group-out gains (positive: the group helped the full model):')
    print(reports['leave_one_out_gain.csv'].to_string(index=False))
    print('\nSingle-group gains (positive: adding just the group helped calendar-only):')
    print(reports['single_group_gain.csv'].to_string(index=False))
    print(f'\nSaved results to: {output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
