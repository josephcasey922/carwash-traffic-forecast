"""One GLOBAL XGBoost model with optional as-of median input features.

Variants keep identical tuned Poisson XGBoost hyperparameters, training window,
archived weather and storm features. Baseline columns are computed OUT-OF-TIME
at each individual training example's original issuance in baseline_features.py.
Never fit the baseline on a future evaluation fold and predict its own training
labels: that would create in-sample baseline features and unfair comparisons.

This experiment is offline, paired, resumable, and does not modify production.
May/June validation was already used for model selection, NOT a final test.
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
import experiments
import model_tuning as tuning
import storm_features as storm
import storm_loss_experiment as sl
import washcast as core

KEYS = list(storm.KEYS)
COLS = ['historical_median_asof', 'weather_median_asof']
VARIANTS = ('xgboost_poisson_storm',
            'xgboost_poisson_storm_histmedian',
            'xgboost_poisson_storm_histmedian_weathermedian',
            'baseline', 'weather_median')
REFERENCE = 'xgboost_poisson_storm'
EXTRA_BY_VARIANT = {
    'xgboost_poisson_storm': (),
    'xgboost_poisson_storm_histmedian': (COLS[0],),
    'xgboost_poisson_storm_histmedian_weathermedian': tuple(COLS),
}


def load_data(examples_path: str, storm_path: str, baseline_path: str,
              window_days: int) -> pd.DataFrame:
    """Validate exact key equality and baseline cache provenance."""
    data = sl.load_enriched(examples_path, storm_path)
    m_path = Path(baseline_path)
    meta_file = m_path.with_suffix('.meta.json')
    if not meta_file.exists():
        raise ValueError(f'No baseline cache metadata at {meta_file}; run baseline_features.py first')
    meta = json.loads(meta_file.read_text(encoding='utf-8'))
    if (meta.get('schema_version') != 1 or meta.get('window_days') != window_days
            or meta.get('min_train_rows') != 200 or meta.get('min_train_origins') != 8):
        raise ValueError('Baseline feature cache configuration incompatible with experiment')
    if meta.get('examples_fingerprint') != tuning.source_fingerprint(examples_path):
        raise ValueError('Baseline feature cache built from a DIFFERENT examples file; regenerate it')
    df = pd.read_csv(m_path, dtype={'location_id': str})
    missing = set(KEYS + COLS) - set(df.columns)
    if missing:
        raise ValueError(f'Baseline cache missing columns: {sorted(missing)}')
    for col in ('issued_at', 'timestamp'):
        df[col] = pd.to_datetime(df[col], utc=True, errors='raise')
    if df.duplicated(KEYS).any():
        raise ValueError('Duplicate keys in baseline cache')
    df = df[KEYS + COLS].copy()
    for col in COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if (df[col].dropna() < 0).any() or np.isinf(df[col].to_numpy(float)).any():
            raise ValueError(f'Invalid baseline feature {col}')
    out = data.merge(df, on=KEYS, how='outer', indicator=True, validate='one_to_one')
    if len(out) != len(data) or not out._merge.eq('both').all():
        raise ValueError('Baseline features do not match the example rows exactly')
    if not (out[COLS[0]].isna() == out[COLS[1]].isna()).all():
        raise ValueError('Baseline feature columns have mismatched missing rows')
    return out.drop(columns='_merge')


def features(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    x = sl.inputs(df, 'storm')
    for col in EXTRA_BY_VARIANT[variant]:
        x[col] = df[col].astype(float)
    return x


def transformer(variant: str) -> ColumnTransformer:
    if variant == REFERENCE:
        # Preserve EXACT preprocessor from storm_loss_experiment.py.
        return sl.transformer('storm')
    nums = experiments.numeric_cols(True) + list(storm.FEATURES) + list(EXTRA_BY_VARIANT[variant])
    return ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), list(core.CAT)),
        ('num', Pipeline([('fill', SimpleImputer(strategy='median', keep_empty_features=True)),
                          ('scale', StandardScaler())]), nums),
    ], sparse_threshold=0)


def selected_xgboost_overrides(configs: str) -> dict:
    overrides, ids = sl.tuned_overrides(configs)
    if 'xgboost' not in ids:
        raise ValueError('Selected tuning configurations do not include XGBoost')
    return overrides['xgboost']


def _paths(out: Path, issue: pd.Timestamp, model: str) -> Path:
    return out / 'fold_predictions' / issue.strftime('%Y%m%dT%H%M%S%z') / (model + '.csv')


def _save(out: Path, issue: pd.Timestamp, model: str, frame: pd.DataFrame):
    path = _paths(out, issue, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.csv.tmp')
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def manifest(args, issues):
    return {
        'schema_version': 1,
        'files': {name: tuning.source_fingerprint(getattr(args, name))
                  for name in ('examples','storm_features','baseline_features','selected_configs')},
        'window_days': args.window_days,
        'validation_from': args.validation_from, 'validation_to': args.validation_to,
        'issuances': [str(t) for t in issues], 'variants': list(VARIANTS),
        'threads': args.threads,
        'notes': [
            'Development comparison on previously viewed May/June issuances; not an untouched test.',
            'Baseline covariates for EACH training row were generated strictly before that row issuance.',
            'All model variants use identical tuned XGBoost Poisson hyperparameters.',
            'Regression preprocessing learned only on the outer fold train rows.',
            'As-issued forecast weather only for target features; reconstructed past weather is approximate.',
            'Forecasts issued on adjacent days overlap; hourly evaluation rows are correlated.',
        ],
    }


def _prepare_output(out: Path, spec: dict):
    out.mkdir(parents=True, exist_ok=True)
    p = out / 'experiment_manifest.json'
    if p.exists():
        if json.loads(p.read_text(encoding='utf-8')) != spec:
            raise ValueError('Output directory belongs to a different run; select a new --output')
    elif list(out.glob('fold_predictions/**/*.csv')):
        raise ValueError('Cannot resume predictions without an experiment manifest')
    else:
        p.write_text(json.dumps(spec, indent=2), encoding='utf-8')


def _pairwise(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    ref = df.loc[df.model.eq(REFERENCE), KEYS + ['carwashes', 'absolute_error']]
    if ref.empty or ref.duplicated(KEYS).any():
        raise ValueError('Reference predictions missing/duplicated')
    rows = []
    for other in (v for v in VARIANTS if v != REFERENCE):
        challenge = df.loc[df.model.eq(other), KEYS + groups + ['carwashes', 'absolute_error']]
        aligned = ref.merge(challenge, on=KEYS, how='outer', validate='one_to_one',
                            suffixes=('_ref', '_new'), indicator=True)
        if not aligned._merge.eq('both').all() or not np.allclose(aligned.carwashes_ref, aligned.carwashes_new):
            raise ValueError(f'Paired target rows differ for {other}')
        split = aligned.groupby(groups, dropna=False, sort=True) if groups else [((), aligned)]
        for key, part in split:
            key = key if isinstance(key, tuple) else (key,)
            record = dict(zip(groups, key))
            record.update({'challenger': other, 'observations': len(part),
                           'reference_mae': float(part.absolute_error_ref.mean()),
                           'challenger_mae': float(part.absolute_error_new.mean()),
                           'gain_vs_original_xgboost': float((part.absolute_error_ref - part.absolute_error_new).mean())})
            rows.append(record)
    return pd.DataFrame(rows)


def run(args):
    if args.window_days < 1 or args.threads < 1:
        raise ValueError('Positive --window-days and --threads required')
    # Let the existing XGBoost estimator module provide its own friendly dependency check.
    import model_benchmark as bm
    bm.require_libraries(('xgboost',))
    overrides = selected_xgboost_overrides(args.selected_configs)
    data = load_data(args.examples, args.storm_features, args.baseline_features, args.window_days)
    issues = tuning.select_validation_issues(data, args.validation_from,
                                              args.validation_to, args.validation_origins)
    out = Path(args.output)
    _prepare_output(out, manifest(args, issues))

    for issue in issues:
        to_do = [v for v in VARIANTS if not _paths(out, issue, v).exists()]
        if not to_do:
            print(f'Resumed {issue}: all variants already complete', flush=True)
            continue
        train, test = backtest.fold_sets(data, issue, args.window_days)
        if len(train) < 200 or train.issued_at.nunique() < 8 or test.empty:
            raise ValueError(f'Insufficient safe training data for {issue}')
        if (train.timestamp >= issue).any() or (train.issued_at >= issue).any():
            raise RuntimeError('Future label present in training fold')
        if test[COLS].isna().any().any():
            raise ValueError(f'Missing as-of baseline features for validation issuance {issue}')
        test = experiments.add_regimes(test)
        test['issue_date'] = test.issued_at.dt.strftime('%Y-%m-%d')
        y = train.carwashes.to_numpy(dtype=float)
        prepared = {}
        for variant in to_do:
            if variant not in EXTRA_BY_VARIANT:
                continue
            proc = transformer(variant)
            xtrain = proc.fit_transform(features(train, variant))
            xtest = proc.transform(features(test, variant))
            prepared[variant] = (xtrain, xtest)
        for variant in to_do:
            if variant == 'baseline':
                prediction = test[COLS[0]].to_numpy(dtype=float)
            elif variant == 'weather_median':
                prediction = test[COLS[1]].to_numpy(dtype=float)
            else:
                xtrain, xtest = prepared[variant]
                estimator = sl.make_estimator('xgboost','poisson',threads=args.threads,overrides=overrides)
                estimator.fit(xtrain, y)
                prediction = sl.predict(estimator,'xgboost','poisson',xtest)
                del estimator
            if not np.isfinite(prediction).all() or (prediction < 0).any():
                raise RuntimeError(f'Invalid predictions for {variant}')
            fields = KEYS + ['local_date','lead_day','issue_date','carwashes','weather_regime',
                             'storm_hours_since_rain']
            if 'inferred_zero' in test.columns:
                fields.append('inferred_zero')
            result = test[fields].copy()
            result['model'] = variant
            result['prediction'] = np.round(prediction, 6)
            result['absolute_error'] = abs(result.carwashes - result.prediction)
            _save(out, issue, variant, result)
            print(f'{issue.date()} {variant:<50} MAE={result.absolute_error.mean():.3f}', flush=True)

    paths = [_paths(out, issue, name) for issue in issues for name in VARIANTS]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f'Incomplete evaluation; {len(missing)} expected fold files missing')
    preds = pd.concat([pd.read_csv(p, dtype={'location_id': str}) for p in paths], ignore_index=True)
    preds['issued_at'] = pd.to_datetime(preds.issued_at, utc=True)
    preds['timestamp'] = pd.to_datetime(preds.timestamp, utc=True)
    preds['storm_recency_bin'] = (pd.cut(preds.storm_hours_since_rain,
        bins=[0,3,6,12,24,72,74],labels=['1-3h','4-6h','7-12h','13-24h','25-72h','>72h'],include_lowest=True)
        .astype(object).fillna('missing history'))
    report = {
        'overall_metrics.csv': experiments.grouped_metrics(preds,['model']).sort_values('mae'),
        'by_weather_regime.csv': experiments.grouped_metrics(preds,['model','weather_regime']),
        'by_issuance.csv': experiments.grouped_metrics(preds,['model','issue_date']),
        'by_location.csv': experiments.grouped_metrics(preds,['model','location_id']),
        'by_lead_day.csv': experiments.grouped_metrics(preds,['model','lead_day']),
        'by_storm_recency.csv': experiments.grouped_metrics(preds,['model','storm_recency_bin']),
        'gain_vs_original.csv': _pairwise(preds, []),
        'gain_vs_original_by_regime.csv': _pairwise(preds, ['weather_regime']),
        'gain_vs_original_by_issuance.csv': _pairwise(preds, ['issue_date']),
    }
    if 'inferred_zero' in preds.columns:
        report['by_inferred_zero.csv'] = experiments.grouped_metrics(preds,['model','inferred_zero'])
    for name, part in report.items():
        part.to_csv(out / name, index=False)
    preds.to_csv(out / 'all_predictions.csv', index=False)
    print('\nBaseline-informed global model results (development only):', flush=True)
    print(report['overall_metrics.csv'].to_string(index=False), flush=True)
    print(f'\nSaved to {out.resolve()}', flush=True)
    return preds, report


def make_parser():
    p = argparse.ArgumentParser(description='One-model baseline-informed global XGBoost/Poisson comparison')
    p.add_argument('--examples', required=True)
    p.add_argument('--storm-features', required=True)
    p.add_argument('--baseline-features', required=True)
    p.add_argument('--selected-configs', required=True)
    p.add_argument('--window-days', type=int, default=365)
    p.add_argument('--validation-from', default='2026-05-01')
    p.add_argument('--validation-to', default='2026-06-30')
    p.add_argument('--validation-origins', type=int, default=6)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--output', default='baseline_informed_may_june')
    return p


def main(argv=None):
    run(make_parser().parse_args(argv))


if __name__ == '__main__':
    main()
