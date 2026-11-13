"""Build STRICTLY as-of historical/weather median features for WashCast.

One cached pair of baseline forecasts per (issued_at, location_id, timestamp).
The feature for a historical training row is computed at THAT ROW'S issuance,
not using a baseline fit on the later outer evaluation fold. This prevents
in-sample/target leakage when a baseline prediction becomes an ML input.

Features use past labeled as-of forecast examples only; no weather API calls.
Weather-adjusted median matches model_tuning.WeatherAdjustedMedian.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import backtest
import experiments
import model_tuning as tuning
import washcast as core

KEYS = ['issued_at', 'location_id', 'timestamp']
FEATURES = ('historical_median_asof', 'weather_median_asof')
SCHEMA_VERSION = 1



def vector_median_predict(model: core.Baseline, rows: pd.DataFrame) -> np.ndarray:
    """Exactly replicate Baseline.predict's three fallback levels, in bulk.

    Using .reindex on median lookup tables avoids millions of Python per-row
    MultiIndex lookups while generating the historical feature cache.
    """
    if rows.empty:
        return np.empty(0, dtype=float)
    exact_keys = pd.MultiIndex.from_frame(rows[['location_id', 'dow', 'hour']])
    site_hour_keys = pd.MultiIndex.from_frame(rows[['location_id', 'hour']])
    exact = model.exact.reindex(exact_keys).to_numpy(dtype=float)
    site_hour = model.site_hour.reindex(site_hour_keys).to_numpy(dtype=float)
    site = model.site.reindex(rows['location_id']).to_numpy(dtype=float)
    return np.where(np.isfinite(exact), exact,
                    np.where(np.isfinite(site_hour), site_hour,
                             np.where(np.isfinite(site), site, float(model.global_median))))


def paired_baseline_predict(past: pd.DataFrame, requested: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Same two predictions as WeatherAdjustedMedian, with vectorized medians.

    Both baselines are fit on `past` only; historical residual adjustments use
    archived-forecast weather regimes and only actual labels within `past`.
    """
    median = core.Baseline().fit(past)
    historical = vector_median_predict(median, requested)
    past_regimes = experiments.add_regimes(past)
    residuals = (past_regimes.carwashes.to_numpy(dtype=float)
                 - vector_median_predict(median, past_regimes))
    past_regimes['residual'] = residuals
    global_adj = {}
    for regime, group in past_regimes.groupby('weather_regime', observed=True):
        n = len(group)
        global_adj[regime] = n / (n + 300) * float(group.residual.median())
    local_adj = {}
    for (site, regime), group in past_regimes.groupby(['location_id', 'weather_regime'], observed=True):
        n = len(group)
        w = n / (n + 160)
        local_adj[(str(site), regime)] = (
            w * float(group.residual.median()) + (1 - w) * global_adj.get(regime, 0.))
    requested_regimes = experiments.add_regimes(requested)
    adjustment = np.asarray([
        local_adj.get((str(site), regime), global_adj.get(regime, 0.))
        for site, regime in zip(requested_regimes.location_id, requested_regimes.weather_regime)
    ], dtype=float)
    return historical, np.maximum(0., historical + adjustment)


def derive_features(examples: pd.DataFrame, *, window_days: int = 365,
                    min_train_rows: int = 200, min_train_origins: int = 8,
                    progress_every: int = 40) -> pd.DataFrame:
    """For each issuance U, fit baseline using only labels timestamp < U.

    The same rolling window, cutoff, fallback and shrinkage are used for
    training-example baseline features and held-out evaluation forecasts.
    Issuances with too little prior history receive NaN (imputed using only
    the outer fold's training rows by the ML preprocessing pipeline).
    """
    if window_days < 1 or min_train_rows < 1 or min_train_origins < 1:
        raise ValueError('window_days, min_train_rows and min_train_origins must be positive')
    if not set(KEYS + ['carwashes']).issubset(examples):
        raise ValueError('Missing example keys/target label')
    if examples.duplicated(KEYS).any():
        raise ValueError('Duplicate example keys')
    if not isinstance(examples['issued_at'].dtype, pd.DatetimeTZDtype) or not isinstance(examples['timestamp'].dtype, pd.DatetimeTZDtype):
        raise ValueError('Example timestamp columns must be timezone-aware datetimes')
    if (examples.timestamp <= examples.issued_at).any():
        raise ValueError('Target timestamps must follow their forecast issuance')
    rows = []
    issues = sorted(examples.issued_at.unique())
    for number, issue in enumerate(issues, 1):
        past, requested = backtest.fold_sets(examples, issue, window_days)
        n = len(requested)
        if not n:
            continue
        part = requested[KEYS].copy()
        if len(past) < min_train_rows or past.issued_at.nunique() < min_train_origins:
            for name in FEATURES:
                part[name] = np.nan
        else:
            if (past.issued_at >= issue).any() or (past.timestamp >= issue).any():
                raise RuntimeError('Baseline feature attempted to use a future label')
            hist, weather = paired_baseline_predict(past, requested)
            part[FEATURES[0]] = hist
            part[FEATURES[1]] = weather
            for name in FEATURES:
                values = part[name].to_numpy(dtype=float)
                if not np.isfinite(values).all() or (values < 0).any() or len(values) != n:
                    raise RuntimeError(f'Invalid {name} at {issue}')
        rows.append(part)
        if progress_every and (number % progress_every == 0 or number == len(issues)):
            print(f'  Historical baseline features: {number:,}/{len(issues):,} issuances', flush=True)
    if not rows:
        raise ValueError('No issuances to process')
    result = pd.concat(rows, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    if len(result) != len(examples) or result.duplicated(KEYS).any():
        raise RuntimeError('Baseline feature generation dropped or duplicated rows')
    return result


def meta_path(path: str | Path) -> Path:
    return Path(path).with_suffix('.meta.json')


def main(argv=None):
    p = argparse.ArgumentParser(description='Create as-of median features for each historical WashCast issuance')
    p.add_argument('--examples', required=True)
    p.add_argument('--window-days', type=int, default=365)
    p.add_argument('--min-train-rows', type=int, default=200)
    p.add_argument('--min-train-origins', type=int, default=8)
    p.add_argument('--output', default='baseline_features_v1/baseline_features.csv')
    args = p.parse_args(argv)
    examples = backtest.load_examples(args.examples)
    print(f'Computing as-of baseline features for {len(examples):,} example rows', flush=True)
    calculated = derive_features(examples, window_days=args.window_days,
                                 min_train_rows=args.min_train_rows,
                                 min_train_origins=args.min_train_origins)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    calculated.to_csv(tmp, index=False, float_format='%.10g')
    tmp.replace(path)
    meta = {
        'schema_version': SCHEMA_VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'examples_fingerprint': tuning.source_fingerprint(args.examples),
        'window_days': args.window_days,
        'min_train_rows': args.min_train_rows,
        'min_train_origins': args.min_train_origins,
        'rows': len(calculated), 'issuances': int(examples.issued_at.nunique()),
        'unavailable_rows': int(calculated[FEATURES[0]].isna().sum()),
        'features': list(FEATURES),
        'notes': [
            'Historical feature values calculated separately at every row issuance.',
            'Past training data require BOTH issued_at and target timestamp < feature issuance.',
            'Weather baseline uses as-issued forecasts in historical examples; no target realized weather.',
            'Early issuances without sufficient history receive missing values (train-fold imputation).',
            'Features use the same rolling window as the downstream ML evaluation.',
            'Historical as-issued forecasts can repeat the same realized target across issuances.',
        ],
    }
    meta_path(path).write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'Saved {len(calculated):,} rows ({meta["unavailable_rows"]:,} warm-up missing) to {path.resolve()}', flush=True)


if __name__ == '__main__':
    main()
