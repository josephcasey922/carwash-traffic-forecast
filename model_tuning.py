"""Offline, resumable chronological hyperparameter tuning for global WashCast models.

Uses the SAME archived as-issued weather features, transformation, and 365-day
rolling fold logic as model_benchmark.py. It never requests weather or touches
raw traffic data. May/June 2026 is the suggested *development* validation
period; leave later unseen dates out of the search for a forward-time check.

Also scores two non-ML references: the existing historical median and a simple
historically calibrated median + forecast-weather correction. Neither stands
in for an actual operator's forecast, which must be recorded before outcomes.

Each completed candidate x issuance evaluation is committed to SQLite so a
stopped run can resume without refitting completed trials. Selected settings
are exported as JSON; production model artifacts are NOT written here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3

import sklearn
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import backtest
import experiments
import model_benchmark as bm
import washcast as core

REGIMES = tuple(experiments.REGIME_ORDER)
ALGORITHMS = ('boosted', 'lightgbm', 'xgboost', 'catboost')
REFERENCE_MODELS = ('baseline', 'weather_median')
SCHEMA_VERSION = 1


def parse_models(spec: str) -> tuple[str, ...]:
    models = tuple(s.strip().lower() for s in spec.split(',') if s.strip())
    if not models or len(set(models)) != len(models) or not set(models).issubset(ALGORITHMS):
        raise ValueError('Choose unique tunable models from: ' + ','.join(ALGORITHMS))
    return models


def source_fingerprint(path: str | Path) -> dict:
    source = Path(path).resolve()
    digest = hashlib.sha256()
    with source.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(source), 'bytes': source.stat().st_size, 'sha256': digest.hexdigest()}


def select_validation_issues(df: pd.DataFrame, start: str, end: str,
                             origins: int) -> list[pd.Timestamp]:
    """Chronologically spaced issuances; no picking dates based on observed errors."""
    if origins < 1:
        raise ValueError('--validation-origins must be positive')
    first, last = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    if first > last:
        raise ValueError('--validation-from cannot exceed --validation-to')
    available = sorted(x for x in df.issued_at.unique()
                       if first <= pd.Timestamp(x).date() <= last)
    if len(available) < origins:
        raise ValueError(f'Only {len(available)} forecast issuances available in '
                         f'{start}..{end}; requested {origins}')
    positions = np.linspace(0, len(available) - 1, origins).round().astype(int)
    if len(set(positions)) != origins:
        raise RuntimeError('Could not pick distinct validation dates')
    return [available[int(i)] for i in positions]


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return round(float(np.exp(rng.uniform(np.log(low), np.log(high)))), 5)


def trial_specs(models: tuple[str, ...], candidates_per_model: int,
                seed: int) -> list[dict]:
    """Default configuration + deterministic randomized candidates per model.

    These parameter ranges are intentionally modest to limit CPU/RAM demand.
    Each model has an independent seed so choosing a subset does not alter the
    candidates for the others.
    """
    if candidates_per_model < 1 or candidates_per_model > 100:
        raise ValueError('Choose 1..100 candidates per model (including the default)')
    trials = []
    for name in models:
        trials.append({'model': name, 'candidate_id': 'default', 'overrides': {}})
        rng = np.random.default_rng(seed + sum((i + 1) * ord(c) for i, c in enumerate(name)))
        used = {json.dumps({}, sort_keys=True)}
        i = 1
        while i < candidates_per_model:
            if name == 'boosted':
                override = {'max_iter': int(rng.integers(90, 301)),
                            'max_leaf_nodes': int(rng.integers(12, 40)),
                            'learning_rate': _log_uniform(rng, .035, .13),
                            'l2_regularization': round(float(rng.uniform(.05, 4.0)), 3),
                            'min_samples_leaf': int(rng.integers(12, 45))}
            elif name == 'lightgbm':
                override = {'n_estimators': int(rng.integers(140, 421)),
                            'num_leaves': int(rng.integers(12, 46)),
                            'learning_rate': _log_uniform(rng, .035, .13),
                            'min_child_samples': int(rng.integers(18, 81)),
                            'reg_lambda': round(float(rng.uniform(.1, 7.0)), 3)}
            elif name == 'xgboost':
                override = {'n_estimators': int(rng.integers(150, 421)),
                            'max_depth': int(rng.integers(3, 8)),
                            'learning_rate': _log_uniform(rng, .035, .13),
                            'min_child_weight': round(float(rng.uniform(1., 12.)), 3),
                            'subsample': round(float(rng.uniform(.7, 1.)), 3),
                            'colsample_bytree': round(float(rng.uniform(.7, 1.)), 3),
                            'reg_lambda': round(float(rng.uniform(.2, 8.)), 3)}
            elif name == 'catboost':
                override = {'iterations': int(rng.integers(120, 361)),
                            'depth': int(rng.integers(4, 9)),
                            'learning_rate': _log_uniform(rng, .035, .13),
                            'l2_leaf_reg': round(float(rng.uniform(1., 12.)), 3)}
            else:
                raise ValueError(name)
            key = json.dumps(override, sort_keys=True)
            if key in used:
                continue
            used.add(key)
            trials.append({'model': name, 'candidate_id': f'random_{i:02d}', 'overrides': override})
            i += 1
    return trials


def _group_stats(actual: np.ndarray, forecast: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if (len(actual) != len(forecast) or not np.isfinite(forecast).all()
            or (forecast < 0).any() or not np.isfinite(actual).all()):
        raise ValueError('Invalid prediction arrays')
    err = forecast - actual
    return {'n': int(len(actual)), 'sum_abs': float(np.abs(err).sum()),
            'sum_sq': float(np.square(err).sum()), 'sum_signed': float(err.sum()),
            'sum_actual': float(actual.sum())}


def score_fold(test: pd.DataFrame, predicted: np.ndarray) -> dict:
    actual = test.carwashes.to_numpy(dtype=float)
    all_stats = _group_stats(actual, predicted)
    regions = {}
    for regime in REGIMES:
        mask = test.weather_regime.eq(regime).to_numpy()
        regions[regime] = _group_stats(actual[mask], np.asarray(predicted)[mask])
    return {'all': all_stats, 'by_regime': regions}


class WeatherAdjustedMedian:
    """Explainable median baseline + learned additive weather-regime offsets.

    Weather categories use the forecast available at each archived issuance.
    Historical residual corrections are learned strictly from the train fold.
    Location-specific corrections shrink toward overall regime corrections.
    No manual/weather-oracle rules and no future observed rainfall are used.
    """
    def __init__(self, local_shrink: float = 160, global_shrink: float = 300):
        self.local_shrink = local_shrink
        self.global_shrink = global_shrink

    def fit(self, train: pd.DataFrame):
        self.median = core.Baseline().fit(train)
        past = experiments.add_regimes(train)
        past['residual'] = past.carwashes.to_numpy(dtype=float) - self.median.predict(past)
        self.global_adjustments = {}
        for regime, group in past.groupby('weather_regime', observed=True):
            n = len(group)
            self.global_adjustments[regime] = (n / (n + self.global_shrink)) * float(group.residual.median())
        self.local_adjustments = {}
        for (site, regime), group in past.groupby(['location_id', 'weather_regime'], observed=True):
            n = len(group)
            pooled = self.global_adjustments.get(regime, 0.)
            w = n / (n + self.local_shrink)
            self.local_adjustments[(str(site), regime)] = (w * float(group.residual.median())
                                                           + (1 - w) * pooled)
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        enriched = experiments.add_regimes(test)
        baseline = self.median.predict(enriched)
        correction = np.array([
            self.local_adjustments.get((str(site), regime),
                self.global_adjustments.get(regime, 0.))
            for site, regime in zip(enriched.location_id, enriched.weather_regime)
        ], dtype=float)
        return np.maximum(0., baseline + correction)


def build_manifest(args, trials: list[dict], selected: list[pd.Timestamp],
                   library_versions: dict | None = None) -> dict:
    return {
        'schema_version': SCHEMA_VERSION,
        'library_versions': library_versions or {},
        'source': source_fingerprint(args.examples),
        'window_days': args.window_days,
        'models': parse_models(args.models),
        'validation_from': args.validation_from,
        'validation_to': args.validation_to,
        'selected_issuances': [str(s) for s in selected],
        'validation_origins': args.validation_origins,
        'candidates_per_model': args.candidates_per_model,
        'seed': args.seed,
        'threads': args.threads,
        'weather_median': not args.no_weather_median,
        'trials': trials,
        'note': 'All model selection uses these development validation dates, not an untouched test set.',
    }


def prepare_output(folder: Path, manifest: dict) -> sqlite3.Connection:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'tuning_manifest.json'
    if path.exists():
        existing = json.loads(path.read_text(encoding='utf-8'))
        if existing != json.loads(json.dumps(manifest)):
            raise ValueError(f'{folder} contains a different tuning run. '
                             'Use a new --output directory to avoid mixing results.')
    elif (folder / 'tuning_progress.sqlite').exists():
        raise ValueError('Progress database found without manifest; use a new output directory')
    else:
        temp = folder / 'tuning_manifest.json.tmp'
        temp.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        temp.replace(path)
    db = sqlite3.connect(folder / 'tuning_progress.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS results ('
               'issued_at TEXT NOT NULL, model TEXT NOT NULL, candidate_id TEXT NOT NULL, '
               'payload TEXT NOT NULL, PRIMARY KEY (issued_at, model, candidate_id))')
    db.commit()
    return db


def finished_keys(db: sqlite3.Connection) -> set[tuple[str, str, str]]:
    return set(db.execute('SELECT issued_at, model, candidate_id FROM results'))


def save_result(db: sqlite3.Connection, record: dict):
    with db:
        db.execute('INSERT INTO results VALUES (?, ?, ?, ?)',
                   (record['issued_at'], record['model'], record['candidate_id'],
                    json.dumps(record, sort_keys=True)))


def load_results(db: sqlite3.Connection) -> list[dict]:
    return [json.loads(row[0]) for row in db.execute('SELECT payload FROM results')]


def _pooled(rows: list[dict], key: str, regime: str | None = None) -> dict:
    accum = {'n': 0, 'sum_abs': 0., 'sum_sq': 0., 'sum_signed': 0., 'sum_actual': 0.}
    for row in rows:
        part = row[key]['all'] if regime is None else row[key]['by_regime'][regime]
        for metric in accum:
            accum[metric] += part[metric]
    n = accum['n']
    return {'observations': n,
            'mae': accum['sum_abs'] / n if n else None,
            'rmse': (accum['sum_sq'] / n)**.5 if n else None,
            'wape_percent': 100 * accum['sum_abs'] / accum['sum_actual'] if accum['sum_actual'] > 0 else None,
            'signed_bias': accum['sum_signed'] / n if n else None}


def summarize(rows: list[dict], n_issues: int, trials: list[dict],
              output: Path) -> tuple[pd.DataFrame, dict]:
    raw = []
    for r in rows:
        stats = r['scores']['all']
        raw.append({'model': r['model'], 'candidate_id': r['candidate_id'],
                    'issued_at': r['issued_at'], 'train_rows': r['train_rows'],
                    'latest_training_target': r['latest_training_target'],
                    'observations': stats['n'], 'mae': stats['sum_abs'] / stats['n'],
                    'signed_bias': stats['sum_signed'] / stats['n'],
                    **{f'mae_{i}': (r['scores']['by_regime'][regime]['sum_abs'] /
                         r['scores']['by_regime'][regime]['n']
                         if r['scores']['by_regime'][regime]['n'] else np.nan)
                       for i, regime in enumerate(REGIMES)}})
    folds = pd.DataFrame(raw).sort_values(['model','candidate_id','issued_at'])
    folds.to_csv(output / 'tuning_fold_scores.csv', index=False)
    records = []
    for (model, candidate), part in sorted(_group_trials(rows).items()):
        total = _pooled(part, 'scores')
        base = {'model': model, 'candidate_id': candidate,
                'completed_origins': len(part),
                'mean_issue_mae': float(np.mean([r['scores']['all']['sum_abs'] /
                                r['scores']['all']['n'] for r in part])),
                'pooled_mae': total['mae'], 'rmse': total['rmse'],
                'signed_bias': total['signed_bias'], 'wape_percent': total['wape_percent'],
                'observations': total['observations']}
        for regime, col in zip(REGIMES, ('rain','post_rain','dry')):
            m = _pooled(part, 'scores', regime)
            base[f'{col}_mae'] = m['mae']
            base[f'{col}_observations'] = m['observations']
        # Incomplete candidates are visible but NEVER eligible for selection.
        base['eligible'] = len(part) == n_issues
        records.append(base)
    summary = pd.DataFrame(records).sort_values(['model','mean_issue_mae'])
    summary.to_csv(output / 'tuning_summary.csv', index=False)
    winners = {}
    lookup = {(t['model'], t['candidate_id']): t for t in trials}
    for name in ALGORITHMS:
        eligible = summary[(summary.model == name) & summary.eligible]
        if eligible.empty:
            continue
        best = eligible.sort_values(['mean_issue_mae','candidate_id']).iloc[0]
        spec = lookup[(name, best.candidate_id)]
        winners[name] = {'candidate_id': spec['candidate_id'],
                         'parameter_overrides': spec['overrides'],
                         'mean_issue_mae': float(best.mean_issue_mae),
                         'pooled_mae': float(best.pooled_mae),
                         'rain_mae': None if pd.isna(best.rain_mae) else float(best.rain_mae),
                         'post_rain_mae': None if pd.isna(best.post_rain_mae) else float(best.post_rain_mae),
                         'dry_mae': None if pd.isna(best.dry_mae) else float(best.dry_mae)}
    # Also give the reference scores, if complete, in selected config report.
    references = {}
    for name in REFERENCE_MODELS:
        ref = summary[(summary.model == name) & summary.eligible]
        if len(ref):
            rec = ref.iloc[0]
            references[name] = {'mean_issue_mae': float(rec.mean_issue_mae),
                                'pooled_mae': float(rec.pooled_mae),
                                'rain_mae': None if pd.isna(rec.rain_mae) else float(rec.rain_mae),
                                'post_rain_mae': None if pd.isna(rec.post_rain_mae) else float(rec.post_rain_mae),
                                'dry_mae': None if pd.isna(rec.dry_mae) else float(rec.dry_mae)}
    selected = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'selection_metric': 'mean_issue_mae (each validation issuance equal weight)',
                'validation_only': True, 'window_days': None,
                'selected_models': winners, 'reference_models': references,
                'usage': 'Pass parameter_overrides to model_benchmark.make_estimator(name, threads).set_params(**overrides) before fitting, on held-out future issuances.',
                'warnings': ['May/June are DEVELOPMENT validation, not an untouched test.',
                             'Check rainy and post-rain MAE, dry MAE and issue-by-issue stability before selecting a deployment model.',
                             'Weather-adjusted median is a statistical heuristic, not a recorded expert forecast.',
                             'Overlapping forecast horizons mean issuance scores are correlated.']}
    return summary, selected


def _group_trials(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped = {}
    for row in rows:
        grouped.setdefault((row['model'], row['candidate_id']), []).append(row)
    return grouped


def run(args) -> tuple[pd.DataFrame, dict]:
    models = parse_models(args.models)
    if args.window_days < 1 or args.threads < 1:
        raise ValueError('Window and thread counts must be positive')
    installed = bm.require_libraries(models)
    versions = {**installed, 'scikit-learn': sklearn.__version__,
                'numpy': np.__version__, 'pandas': pd.__version__}
    df = experiments.add_season_features(backtest.load_examples(args.examples))
    selected = select_validation_issues(df, args.validation_from,
                                         args.validation_to, args.validation_origins)
    trials = trial_specs(models, args.candidates_per_model, args.seed)
    refs = [dict(model='baseline', candidate_id='reference', overrides={})]
    if not args.no_weather_median:
        refs.append(dict(model='weather_median', candidate_id='reference', overrides={}))
    tasks = refs + trials
    manifest = build_manifest(args, trials, selected, versions)
    folder = Path(args.output)
    db = prepare_output(folder, manifest)
    try:
        done = finished_keys(db)
        for issue in selected:
            issue_str = str(issue)
            pending = [t for t in tasks if (issue_str, t['model'], t['candidate_id']) not in done]
            if not pending:
                print(f'Resumed: {issue_str} fully completed; skipping', flush=True)
                continue
            train, test = backtest.fold_sets(df, issue, args.window_days)
            if (len(train) < 200 or train.issued_at.nunique() < 8 or test.empty
                    or (train.issued_at >= issue).any() or (train.timestamp >= issue).any()):
                raise ValueError(f'Insufficient safe training data for {issue_str}')
            y = train.carwashes.to_numpy(dtype=float)
            test = experiments.add_regimes(test)
            print(f'Validation {issue_str}: {len(train):,} training, {len(test):,} test rows; '
                  f'{len(pending)} remaining fits', flush=True)
            needed_ml = any(t['model'] in ALGORITHMS for t in pending)
            if needed_ml:
                transform = experiments.make_transformer(True)
                x_train = transform.fit_transform(experiments.inputs(train, True))
                x_test = transform.transform(experiments.inputs(test, True))
                if 'lightgbm' in models:
                    names = transform.get_feature_names_out()
                    x_train_lgbm = pd.DataFrame(x_train, columns=names)
                    x_test_lgbm = pd.DataFrame(x_test, columns=names)
            for t in pending:
                name, candidate = t['model'], t['candidate_id']
                if name == 'baseline':
                    pred = np.maximum(0., core.Baseline().fit(train).predict(test))
                elif name == 'weather_median':
                    pred = WeatherAdjustedMedian().fit(train).predict(test)
                else:
                    estimator = bm.make_estimator(name, args.threads)
                    if t['overrides']:
                        estimator.set_params(**t['overrides'])
                    xi = x_train_lgbm if name == 'lightgbm' else x_train
                    xo = x_test_lgbm if name == 'lightgbm' else x_test
                    estimator.fit(xi, y)
                    pred = bm.response_prediction(estimator, name, xo)
                    del estimator
                score = score_fold(test, pred)
                record = {'issued_at': issue_str, 'model': name, 'candidate_id': candidate,
                          'train_rows': int(len(train)), 'test_rows': int(len(test)),
                          'latest_training_target': str(train.timestamp.max()),
                          'scores': score}
                save_result(db, record)
                done.add((issue_str, name, candidate))
                print(f'  {name:<14} {candidate:<12} MAE={score["all"]["sum_abs"]/score["all"]["n"]:.3f}',
                      flush=True)
        rows = load_results(db)
        summary, selected_models = summarize(rows, len(selected), trials, folder)
        selected_models['window_days'] = args.window_days
        selected_models['validation_issuances'] = [str(s) for s in selected]
        (folder / 'selected_configurations.json').write_text(json.dumps(selected_models, indent=2),
                                                             encoding='utf-8')
        total = len(tasks) * len(selected)
        if len(rows) != total:
            raise RuntimeError(f'{len(rows)}/{total} candidate-issue results complete')
        print('\nComplete — best candidate within each algorithm (development validation):')
        preferred = summary[summary.eligible].sort_values('mean_issue_mae').groupby('model', sort=False).head(1)
        print(preferred[['model','candidate_id','mean_issue_mae','pooled_mae',
                         'rain_mae','post_rain_mae','dry_mae']].sort_values('mean_issue_mae').to_string(index=False))
        print(f'\nResults saved to {folder.resolve()}', flush=True)
        return summary, selected_models
    finally:
        db.close()


def make_parser():
    p = argparse.ArgumentParser(description='Offline, resumable chronological global model tuning')
    p.add_argument('--examples', required=True)
    p.add_argument('--window-days', type=int, default=365)
    p.add_argument('--validation-from', default='2026-05-01',
                   help='Start of development-validation forecast issuances (UTC date)')
    p.add_argument('--validation-to', default='2026-06-30')
    p.add_argument('--validation-origins', type=int, default=6,
                   help='Evenly spaced forecast issuances; count, not consecutive days')
    p.add_argument('--models', default=','.join(ALGORITHMS))
    p.add_argument('--candidates-per-model', type=int, default=5,
                   help='Includes unchanged default; e.g. 5 => 1 default + 4 randomized')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--no-weather-median', action='store_true',
                   help='Skip the interpretable historical median + predicted-weather comparator')
    p.add_argument('--output', default='model_tuning_may_june')
    return p


def main(argv=None):
    args = make_parser().parse_args(argv)
    run(args)


if __name__ == '__main__':
    main()
