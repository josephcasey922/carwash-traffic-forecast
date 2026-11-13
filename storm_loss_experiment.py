"""Offline, rolling-origin experiment: storm features x Poisson/MAE training loss.

All variants predict the SAME as-of targets using strictly earlier labeled
examples. May/June are development validation dates, not untouched test data.
The original boosted+Poisson variant must reproduce the existing benchmark.
Forecast input always comes from archived issuance-specific weather; added storm
features are joined from storm_features.py's separately validated cache.

This module does NOT modify production WashCast, existing examples, or models.
Individual fold/variant predictions are saved so interrupted runs can resume.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import backtest
import experiments
import model_benchmark as bm
import model_tuning as tuning
import storm_features as storm
import washcast as core

KEYS = list(storm.KEYS)
ALGORITHMS = ('boosted', 'xgboost', 'catboost', 'lightgbm')
LOSSES = ('poisson', 'absolute')
FEATURE_SETS = ('original', 'storm')
REFERENCE = 'boosted_poisson_original'


def parse_choices(raw: str, allowed: tuple[str, ...], name: str) -> tuple[str, ...]:
    parts = tuple(p.strip().lower() for p in raw.split(',') if p.strip())
    if not parts or len(set(parts)) != len(parts) or not set(parts).issubset(allowed):
        raise ValueError(f'{name} must contain unique names from {allowed}')
    return parts


def load_enriched(examples_path: str, features_path: str) -> pd.DataFrame:
    examples = backtest.load_examples(examples_path)
    extra = pd.read_csv(features_path, dtype={'location_id': str})
    needed = set(KEYS + list(storm.FEATURES))
    if not needed.issubset(extra.columns):
        raise ValueError(f'Storm feature CSV missing {sorted(needed-set(extra.columns))}')
    extra['issued_at'] = pd.to_datetime(extra['issued_at'], utc=True, errors='raise')
    extra['timestamp'] = pd.to_datetime(extra['timestamp'], utc=True, errors='raise')
    if extra.duplicated(KEYS).any():
        raise ValueError('Storm features include duplicate target keys')
    extra = extra[KEYS + list(storm.FEATURES)].copy()
    for name in storm.FEATURES:
        extra[name] = pd.to_numeric(extra[name], errors='raise')
    matched = examples.merge(extra, on=KEYS, how='outer', validate='one_to_one', indicator=True)
    if not matched._merge.eq('both').all() or len(matched) != len(examples):
        missing = matched._merge.value_counts().to_dict()
        raise ValueError(f'Storm features not an exact key match to examples: {missing}')
    return experiments.add_season_features(matched.drop(columns='_merge'))


def transformer(feature_set: str) -> ColumnTransformer:
    if feature_set == 'original':
        return experiments.make_transformer(True)
    if feature_set != 'storm':
        raise ValueError(feature_set)
    nums = experiments.numeric_cols(True) + list(storm.FEATURES)
    return ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), list(core.CAT)),
        ('num', Pipeline([('fill', SimpleImputer(strategy='median', keep_empty_features=True)),
                          ('scale', StandardScaler())]), nums),
    ], sparse_threshold=0)


def inputs(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if feature_set == 'original':
        return experiments.inputs(df, True)
    names = list(core.CAT) + experiments.numeric_cols(True) + list(storm.FEATURES)
    x = df[names].copy()
    for col in core.CAT:
        x[col] = x[col].astype(str)
    return x


def tuned_overrides(path: str | None) -> tuple[dict, dict]:
    if not path:
        return {}, {}
    source = Path(path)
    document = json.loads(source.read_text(encoding='utf-8'))
    selected = document.get('selected_models', {})
    overrides = {name: value.get('parameter_overrides', {})
                 for name, value in selected.items() if name in ALGORITHMS}
    ids = {name: value.get('candidate_id', 'unknown')
           for name, value in selected.items() if name in ALGORITHMS}
    return overrides, ids


def make_estimator(algorithm: str, loss: str, *, threads: int = 2,
                   overrides: dict | None = None):
    est = bm.make_estimator(algorithm, threads)
    if overrides:
        est.set_params(**overrides)
    options = {
        'boosted': {'poisson': 'poisson', 'absolute': 'absolute_error'},
        'xgboost': {'poisson': 'count:poisson', 'absolute': 'reg:absoluteerror'},
        'catboost': {'poisson': 'Poisson', 'absolute': 'MAE'},
        'lightgbm': {'poisson': 'poisson', 'absolute': 'regression_l1'},
    }
    param = 'loss_function' if algorithm == 'catboost' else ('loss' if algorithm == 'boosted' else 'objective')
    est.set_params(**{param: options[algorithm][loss]})
    return est


def predict(estimator, algorithm: str, loss: str, x) -> np.ndarray:
    if algorithm == 'catboost' and loss == 'poisson':
        values = estimator.predict(x, prediction_type='Exponent')
    else:
        values = estimator.predict(x)
    pred = np.asarray(values, dtype=float).reshape(-1)
    if len(pred) != len(x) or not np.isfinite(pred).all():
        raise RuntimeError(f'{algorithm}/{loss} returned invalid predictions')
    return np.maximum(0., pred)


def build_variants(algorithms: tuple[str, ...], losses: tuple[str, ...],
                   feature_sets: tuple[str, ...]) -> list[dict]:
    result = [{'name': 'baseline', 'algorithm': 'baseline'},
              {'name': 'weather_median', 'algorithm': 'weather_median'}]
    for algorithm in algorithms:
        for loss in losses:
            for feature_set in feature_sets:
                result.append({'name': f'{algorithm}_{loss}_{feature_set}',
                               'algorithm': algorithm, 'loss': loss, 'features': feature_set})
    if not any(v['name'] == REFERENCE for v in result):
        raise ValueError('For paired comparisons include boosted,poisson,original in selections')
    return result


def _fingerprint(path: str | None):
    return tuning.source_fingerprint(path) if path else None


def manifest_for(args, variants: list[dict], issues: list[pd.Timestamp],
                 candidate_ids: dict) -> dict:
    files = {'examples': _fingerprint(args.examples),
             'storm_features': _fingerprint(args.storm_features),
             'selected_configs': _fingerprint(args.selected_configs)}
    return {
        'schema_version': 1,
        'files': files,
        'window_days': args.window_days,
        'validation_from': args.validation_from,
        'validation_to': args.validation_to,
        'selected_issuances': [str(issue) for issue in issues],
        'variants': variants, 'selected_tuning_candidates': candidate_ids,
        'threads': args.threads,
        'runtime': {'pandas': pd.__version__, 'numpy': np.__version__},
        'notes': [
            'Development validation only: May/June dates used for tuning before this experiment.',
            'Fold training requires issued_at and target timestamp earlier than current issue.',
            'Archive weather and derived storm features are strictly as-of each issue.',
            'Original variant uses precisely the established model features and preprocessing.',
            'Absolute-error models produce conditional-median-style predictions; count clipping is at zero.',
            'Issued forecasts share target hours across nearby issuances; uncertainty must account for clustering.',
            'Storm feature cache may depend on retrospectively reconstructed pre-issuance weather.',
        ],
    }


def prepare_output(folder: Path, manifest: dict):
    folder.mkdir(parents=True, exist_ok=True)
    meta = folder / 'experiment_manifest.json'
    if meta.exists():
        if json.loads(meta.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Output directory belongs to a different run; choose a new --output')
    elif any(folder.glob('fold_predictions/**/*csv')):
        raise ValueError('Existing fold predictions without manifest; use new --output')
    else:
        tmp = meta.with_suffix('.tmp')
        tmp.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        tmp.replace(meta)


def store_prediction(output: Path, issue: pd.Timestamp, variant: str, df: pd.DataFrame):
    parent = output / 'fold_predictions' / issue.strftime('%Y%m%dT%H%M%S%z')
    parent.mkdir(parents=True, exist_ok=True)
    dest = parent / (variant + '.csv')
    tmp = dest.with_suffix('.tmp')
    df.to_csv(tmp, index=False)
    tmp.replace(dest)


def stored_file(output: Path, issue: pd.Timestamp, variant: str) -> Path:
    return output / 'fold_predictions' / issue.strftime('%Y%m%dT%H%M%S%z') / (variant + '.csv')


def _report_pair(preds: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    ref = preds[preds.variant == REFERENCE][KEYS + ['actual','absolute_error']]
    if ref.duplicated(KEYS).any():
        raise ValueError('Duplicate reference prediction keys')
    results = []
    for var in sorted(preds.variant.unique()):
        if var == REFERENCE:
            continue
        challenger = preds[preds.variant == var]
        right = challenger[KEYS + groups + ['actual','absolute_error']]
        joined = ref.merge(right, how='outer', on=KEYS, validate='one_to_one',
                           indicator=True, suffixes=('_ref', '_new'))
        if not joined._merge.eq('both').all() or not np.allclose(joined.actual_ref, joined.actual_new):
            raise RuntimeError('Compared variants were evaluated on different target rows')
        for values, subset in (joined.groupby(groups, dropna=False, sort=True)
                               if groups else [((), joined)]):
            values = values if isinstance(values, tuple) else (values,)
            row = dict(zip(groups, values))
            row.update({'challenger': var, 'observations': len(subset),
                        'reference_mae': subset.absolute_error_ref.mean(),
                        'challenger_mae': subset.absolute_error_new.mean(),
                        'gain_vs_boosted_poisson_original': (
                            subset.absolute_error_ref - subset.absolute_error_new).mean()})
            results.append(row)
    return pd.DataFrame(results)


def evaluate(args):
    algorithms = parse_choices(args.models, ALGORITHMS, '--models')
    losses = parse_choices(args.losses, LOSSES, '--losses')
    feature_sets = parse_choices(args.feature_sets, FEATURE_SETS, '--feature-sets')
    if args.window_days < 1 or args.threads < 1:
        raise ValueError('Positive window and thread count required')
    variants = build_variants(algorithms, losses, feature_sets)
    bm.require_libraries(algorithms)
    overrides, ids = tuned_overrides(args.selected_configs)
    data = load_enriched(args.examples, args.storm_features)
    issues = tuning.select_validation_issues(data, args.validation_from,
                                              args.validation_to, args.validation_origins)
    folder = Path(args.output)
    prepare_output(folder, manifest_for(args, variants, issues, ids))

    for issue in issues:
        todo = [v for v in variants if not stored_file(folder, issue, v['name']).exists()]
        if not todo:
            print(f'Resumed: {issue} complete; skipped', flush=True)
            continue
        train, test = backtest.fold_sets(data, issue, args.window_days)
        if (len(train) < 200 or train.issued_at.nunique() < 8 or test.empty or
                (train.timestamp >= issue).any() or (train.issued_at >= issue).any()):
            raise ValueError(f'Unsafe or insufficient training examples for {issue}')
        test = experiments.add_regimes(test)
        test['issue_date'] = test.issued_at.dt.strftime('%Y-%m-%d')
        # Derive separate, shared original/storm transformations fitted on TRAIN only.
        transformed = {}
        for feature_set in feature_sets:
            if not any(v.get('features') == feature_set for v in todo):
                continue
            trans = transformer(feature_set)
            xt = trans.fit_transform(inputs(train, feature_set))
            xv = trans.transform(inputs(test, feature_set))
            names = trans.get_feature_names_out()
            transformed[feature_set] = (xt, xv, names)
        labels = train.carwashes.to_numpy(dtype=float)
        if not np.isfinite(labels).all() or np.any(labels < 0):
            raise ValueError('Invalid training labels')
        for var in todo:
            model = var['algorithm']
            if model == 'baseline':
                forecast = np.maximum(0., core.Baseline().fit(train).predict(test))
            elif model == 'weather_median':
                forecast = tuning.WeatherAdjustedMedian().fit(train).predict(test)
            else:
                xtrain, xtest, names = transformed[var['features']]
                if model == 'lightgbm':
                    xtrain = pd.DataFrame(xtrain, columns=names)
                    xtest = pd.DataFrame(xtest, columns=names)
                est = make_estimator(model, var['loss'], threads=args.threads,
                                     overrides=overrides.get(model))
                est.fit(xtrain, labels)
                forecast = predict(est, model, var['loss'], xtest)
                del est
            cols = KEYS + ['local_date','lead_day','issue_date','carwashes','weather_regime',
                           'storm_hours_since_rain']
            if 'inferred_zero' in test.columns:
                cols.append('inferred_zero')
            result = test[cols].copy().rename(columns={'carwashes': 'actual'})
            result['variant'] = var['name']
            result['prediction'] = np.round(forecast, 6)
            result['absolute_error'] = (result.actual - result.prediction).abs()
            store_prediction(folder, issue, var['name'], result)
            print(f'{issue.date()}  {var["name"]:<29}  MAE={result.absolute_error.mean():.3f}', flush=True)

    paths = [stored_file(folder, issue, var['name']) for issue in issues for var in variants]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise RuntimeError(f'Incomplete run: {len(missing)} fold files missing')
    predictions = pd.concat([pd.read_csv(p, dtype={'location_id': str}) for p in paths], ignore_index=True)
    predictions['issued_at'] = pd.to_datetime(predictions.issued_at, utc=True)
    predictions['timestamp'] = pd.to_datetime(predictions.timestamp, utc=True)
    actual = predictions['actual'].to_numpy(dtype=float)
    if not np.isfinite(actual).all():
        raise RuntimeError('Nonfinite actual values in cached predictions')
    # Recency uses only as-of weather, not target realization.
    since = predictions.storm_hours_since_rain
    predictions['storm_recency_bin'] = pd.cut(since, bins=[0, 3, 6, 12, 24, 72, 74],
          labels=['1–3h', '4–6h', '7–12h', '13–24h', '25–72h', '>72h'], include_lowest=True)
    predictions['storm_recency_bin'] = (predictions['storm_recency_bin'].astype(object)
                                         .fillna('insufficient weather history'))
    reports = {
        'overall_metrics.csv': experiments.grouped_metrics(
            predictions.rename(columns={'variant': 'model', 'actual': 'carwashes'}), ['model']).sort_values('mae'),
        'by_weather_regime.csv': experiments.grouped_metrics(
            predictions.rename(columns={'variant': 'model', 'actual': 'carwashes'}), ['model','weather_regime']),
        'by_issuance.csv': experiments.grouped_metrics(
            predictions.rename(columns={'variant': 'model', 'actual': 'carwashes'}), ['model','issue_date']),
        'by_location.csv': experiments.grouped_metrics(
            predictions.rename(columns={'variant': 'model', 'actual': 'carwashes'}), ['model','location_id']),
        'by_lead_day.csv': experiments.grouped_metrics(
            predictions.rename(columns={'variant': 'model', 'actual': 'carwashes'}), ['model','lead_day']),
        'by_storm_recency.csv': experiments.grouped_metrics(
            predictions.rename(columns={'variant': 'model', 'actual': 'carwashes'}), ['model','storm_recency_bin']),
        'gain_vs_reference.csv': _report_pair(predictions, []),
        'gain_vs_reference_by_regime.csv': _report_pair(predictions, ['weather_regime']),
        'gain_vs_reference_by_issuance.csv': _report_pair(predictions, ['issue_date']),
    }
    if 'inferred_zero' in predictions.columns:
        reports['by_inferred_zero.csv'] = experiments.grouped_metrics(
            predictions.rename(columns={'variant': 'model', 'actual': 'carwashes'}), ['model','inferred_zero'])
    for name, report in reports.items():
        report.to_csv(folder / name, index=False)
    predictions.to_csv(folder / 'all_predictions.csv', index=False)
    print('\nResults on development issuance dates (lower MAE is better):', flush=True)
    print(reports['overall_metrics.csv'].to_string(index=False), flush=True)
    print(f'\nSaved to {folder.resolve()}', flush=True)
    return predictions, reports


def make_parser():
    p = argparse.ArgumentParser(description='Offline WashCast storm-feature and loss objective comparison')
    p.add_argument('--examples', required=True)
    p.add_argument('--storm-features', required=True)
    p.add_argument('--selected-configs', default=None,
                   help='Optional frozen overrides from model_tuning_may_june/selected_configurations.json')
    p.add_argument('--window-days', type=int, default=365)
    p.add_argument('--validation-from', default='2026-05-01')
    p.add_argument('--validation-to', default='2026-06-30')
    p.add_argument('--validation-origins', type=int, default=6,
                   help='Evenly spaced development issuances')
    p.add_argument('--models', default='boosted,xgboost',
                   help='Choose from boosted,xgboost,catboost,lightgbm; boosted required')
    p.add_argument('--losses', default='poisson,absolute',
                   help='Poisson vs absolute-error (conditional-median) training objectives')
    p.add_argument('--feature-sets', default='original,storm')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--output', default='storm_loss_may_june')
    return p


def main(argv=None):
    args = make_parser().parse_args(argv)
    evaluate(args)


if __name__ == '__main__':
    main()
