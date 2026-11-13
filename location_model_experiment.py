"""Compare store-specific and pooled XGBoost models on identical as-of targets.

No new weather API calls. Uses the EXISTING, strictly-as-of storm and median
feature caches and saved tuning settings. Refits at each historical issuance.

This is an exploratory model-development comparison, not an untouched test.
Data availability for one store can be much more limited than across stores;
don't extrapolate an improvement from one store to all other locations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import backtest
import baseline_informed_experiment as bi
import experiments
import model_tuning as tuning
import storm_loss_experiment as sl

VARIANTS = (
    'global_xgboost_storm',
    'local_xgboost_storm',
    'global_xgboost_storm_histmedian',
    'local_xgboost_storm_histmedian',
    'historical_median',
    'weather_median',
)
REFERENCE = 'global_xgboost_storm'
KEYS = list(bi.KEYS)


def model_configuration(name: str) -> tuple[bool, str]:
    """Returns (local training only, feature specification)."""
    if name not in VARIANTS[:4]:
        raise ValueError(f'Not an XGBoost model: {name}')
    return (name.startswith('local_'),
            'xgboost_poisson_storm_histmedian' if 'histmedian' in name
            else bi.REFERENCE)


def prediction_path(out: Path, issue: pd.Timestamp, variant: str) -> Path:
    return out / 'fold_predictions' / pd.Timestamp(issue).strftime('%Y%m%dT%H%M%S%z') / (variant + '.csv')


def _manifest(args, issuances: list[pd.Timestamp]) -> dict:
    return {
        'schema_version': 1,
        'files': {name: tuning.source_fingerprint(getattr(args, name)) for name in (
            'examples', 'storm_features', 'baseline_features', 'selected_configs')},
        'location': args.location,
        'window_days': args.window_days,
        'validation_from': args.validation_from,
        'validation_to': args.validation_to,
        'issuances': [str(x) for x in issuances],
        'threads': args.threads,
        'models': list(VARIANTS),
        'notes': [
            'This is development validation, not an untouched holdout.',
            'Global and local versions use identical fixed tuned Poisson XGBoost parameters.',
            'Both models predict exactly the same target store and target hours.',
            'Local model fits only target-store rows; global model fits all stores.',
            'Training labels and issuance timestamps strictly precede evaluation issuance.',
            'Median input features were computed from data available before each row issuance.',
            'All weather predictors derive from weather available at the original issuance.',
            'Some weather-history estimates are retrospectively reconstructed before issuance.',
            'Overlapping forecast horizons induce correlated errors.',
        ],
    }


def _init_output(out: Path, spec: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    info = out / 'experiment_manifest.json'
    if info.exists():
        if json.loads(info.read_text(encoding='utf-8')) != spec:
            raise ValueError('Output directory belongs to a different run; choose another --output')
    elif list(out.glob('fold_predictions/**/*.csv')):
        raise ValueError('Cannot resume existing predictions without manifest')
    else:
        info.write_text(json.dumps(spec, indent=2), encoding='utf-8')


def train_predict(train: pd.DataFrame, test: pd.DataFrame, variant: str,
                  overrides: dict, threads: int) -> np.ndarray:
    """Fit transformer and XGBoost ONLY on this fold's training data."""
    if train.empty or test.empty:
        raise ValueError('Empty fold')
    _, features = model_configuration(variant)
    processor = bi.transformer(features)
    x_train = processor.fit_transform(bi.features(train, features))
    x_test = processor.transform(bi.features(test, features))
    y = train.carwashes.to_numpy(dtype=float)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Invalid training targets')
    model = sl.make_estimator('xgboost', 'poisson', threads=threads, overrides=overrides)
    model.fit(x_train, y)
    return sl.predict(model, 'xgboost', 'poisson', x_test)


def paired_gains(predictions: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    ref = predictions.loc[predictions.model.eq(REFERENCE), KEYS + ['carwashes','absolute_error']]
    if ref.empty or ref.duplicated(KEYS).any():
        raise RuntimeError('Reference missing or duplicated')
    rows = []
    for variant in (v for v in VARIANTS if v != REFERENCE):
        challenger = predictions.loc[predictions.model.eq(variant),
            KEYS + groups + ['carwashes','absolute_error']]
        if challenger.empty or challenger.duplicated(KEYS).any():
            raise RuntimeError(f'Challenger missing or duplicated: {variant}')
        both = ref.merge(challenger, on=KEYS, how='outer', suffixes=('_global','_new'),
                         indicator=True, validate='one_to_one')
        if not both._merge.eq('both').all() or not np.allclose(
                both.carwashes_global, both.carwashes_new):
            raise RuntimeError(f'Unpaired comparisons for {variant}')
        for key, part in (both.groupby(groups, dropna=False, sort=True)
                          if groups else [((), both)]):
            key = key if isinstance(key, tuple) else (key,)
            row = dict(zip(groups, key))
            row.update({'challenger': variant, 'observations': len(part),
                        'global_mae': float(part.absolute_error_global.mean()),
                        'challenger_mae': float(part.absolute_error_new.mean()),
                        'gain_vs_global': float((part.absolute_error_global -
                                                 part.absolute_error_new).mean())})
            rows.append(row)
    return pd.DataFrame(rows)


def run(args):
    if not args.location or args.window_days < 1 or args.threads < 1:
        raise ValueError('Provide a store ID, positive window days, and positive threads')
    import model_benchmark as benchmark
    benchmark.require_libraries(('xgboost',))
    overrides = bi.selected_xgboost_overrides(args.selected_configs)
    data = bi.load_data(args.examples, args.storm_features, args.baseline_features,
                        args.window_days)
    if args.location not in set(data.location_id):
        raise ValueError(f'Location {args.location!r} not in examples data')
    site = data.loc[data.location_id.eq(args.location)]
    issues = tuning.select_validation_issues(site, args.validation_from,
                                            args.validation_to, args.validation_origins)
    output = Path(args.output)
    _init_output(output, _manifest(args, issues))

    for issue in issues:
        todo = [name for name in VARIANTS if not prediction_path(output, issue, name).exists()]
        if not todo:
            print(f'{issue.date()}: all six results already saved', flush=True)
            continue
        global_train, global_test = backtest.fold_sets(data, issue, args.window_days)
        local_train = global_train.loc[global_train.location_id.eq(args.location)].copy()
        local_test = global_test.loc[global_test.location_id.eq(args.location)].copy()
        if (len(local_train) < 200 or local_train.issued_at.nunique() < 8 or
                local_test.empty):
            raise ValueError(f'Not enough store-specific history for {args.location} at {issue}: '
                             f'{len(local_train)} rows, {local_train.issued_at.nunique()} issuances')
        if (global_train.timestamp >= issue).any() or (global_train.issued_at >= issue).any():
            raise RuntimeError('Future target/issue in training data')
        if local_test[list(bi.COLS)].isna().any().any():
            raise ValueError('Missing as-of median covariates on test rows')
        local_test = experiments.add_regimes(local_test)
        local_test['issue_date'] = local_test.issued_at.dt.strftime('%Y-%m-%d')
        print(f'{issue.date()} {args.location}  global train {len(global_train):,}, '
              f'local train {len(local_train):,}, test {len(local_test):,}', flush=True)

        for name in todo:
            if name == 'historical_median':
                predicted = local_test[bi.COLS[0]].to_numpy(float)
            elif name == 'weather_median':
                predicted = local_test[bi.COLS[1]].to_numpy(float)
            else:
                local_only, _ = model_configuration(name)
                training = local_train if local_only else global_train
                predicted = train_predict(training, local_test, name, overrides, args.threads)
            if (len(predicted) != len(local_test) or not np.isfinite(predicted).all()
                    or (predicted < 0).any()):
                raise RuntimeError(f'Bad prediction for {name}')
            fields = KEYS + ['local_date','hour','lead_day','issue_date','carwashes',
                             'weather_regime','storm_hours_since_rain']
            if 'inferred_zero' in local_test.columns:
                fields.append('inferred_zero')
            frame = local_test[fields].copy()
            frame['model'] = name
            frame['prediction'] = np.round(predicted, 6)
            frame['absolute_error'] = abs(frame.carwashes - frame.prediction)
            path = prediction_path(output, issue, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix('.csv.tmp')
            frame.to_csv(tmp, index=False)
            tmp.replace(path)
            print(f'  {name:<35} MAE {frame.absolute_error.mean():.3f}', flush=True)

    paths = [prediction_path(output, issue, variant)
             for issue in issues for variant in VARIANTS]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise RuntimeError(f'Missing {len(missing)} expected prediction files')
    pred = pd.concat([pd.read_csv(path, dtype={'location_id': str})
                      for path in paths], ignore_index=True)
    pred['issued_at'] = pd.to_datetime(pred['issued_at'], utc=True)
    pred['timestamp'] = pd.to_datetime(pred['timestamp'], utc=True)
    pred['storm_recency_bin'] = pd.cut(pred['storm_hours_since_rain'],
        bins=[0, 3, 6, 12, 24, 72, 74],
        labels=['1-3h','4-6h','7-12h','13-24h','25-72h','>72h'],
        include_lowest=True).astype(object).fillna('unknown')

    reports = {
        'overall_metrics.csv': experiments.grouped_metrics(pred,['model']).sort_values('mae'),
        'by_weather_regime.csv': experiments.grouped_metrics(pred,['model','weather_regime']),
        'by_issuance.csv': experiments.grouped_metrics(pred,['model','issue_date']),
        'by_storm_recency.csv': experiments.grouped_metrics(pred,['model','storm_recency_bin']),
        'by_lead_day.csv': experiments.grouped_metrics(pred,['model','lead_day']),
        'gain_vs_global.csv': paired_gains(pred, []),
        'gain_vs_global_by_regime.csv': paired_gains(pred, ['weather_regime']),
        'gain_vs_global_by_issuance.csv': paired_gains(pred, ['issue_date']),
    }
    if 'inferred_zero' in pred.columns:
        reports['by_inferred_zero.csv'] = experiments.grouped_metrics(pred,['model','inferred_zero'])
    for filename, table in reports.items():
        table.to_csv(output / filename, index=False)
    pred.to_csv(output / 'all_predictions.csv', index=False)
    print(f'\n{args.location} global vs. store-specific XGBoost (development):')
    print(reports['overall_metrics.csv'].to_string(index=False))
    print(f'\nSaved to {output.resolve()}', flush=True)
    return pred, reports


def make_parser():
    parser = argparse.ArgumentParser(
        description='Single-location vs global model: same site, dates, weather, and tuned settings')
    parser.add_argument('--examples', required=True)
    parser.add_argument('--storm-features', required=True)
    parser.add_argument('--baseline-features', required=True)
    parser.add_argument('--selected-configs', required=True)
    parser.add_argument('--location', required=True, help='Store ID, e.g. AL002')
    parser.add_argument('--window-days', type=int, default=365)
    parser.add_argument('--validation-from', default='2026-05-01')
    parser.add_argument('--validation-to', default='2026-06-30')
    parser.add_argument('--validation-origins', type=int, default=6)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', default='location_model_AL002_may_june')
    return parser


def main(argv=None):
    run(make_parser().parse_args(argv))


if __name__ == '__main__':
    main()
