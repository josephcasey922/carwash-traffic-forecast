"""Leakage-aware, offline storm features for WashCast's saved as-of examples.

Uses historical reconstructed weather ONLY at/before each forecast issuance,
and that issuance's archived forecast for weather AFTER issuance. In particular,
no realized future rain, and no weather from a later forecast issuance, enters
any feature. This is a *separate cache*, so existing training scripts and CSVs
are not modified.

Note: reconstructed historical weather is a proxy for measurements available
at issuance, not proof that the same measurements existed in real time.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import backtest
import washcast as core
import asof

KEYS = ['issued_at', 'location_id', 'timestamp']
RAIN_THRESHOLD_MM = 0.5
FEATURES = (
    'storm_rain_prev3h_mm', 'storm_rain_prev6h_mm', 'storm_rain_prev12h_mm',
    'storm_peak_rain_prev12h_mm', 'storm_hours_since_rain',
    'storm_last_event_duration_h', 'storm_rain_ended_this_hour',
    'storm_cloud_change_3h', 'storm_temperature_change_3h',
)


def derive_hourly_features(grid: pd.DataFrame, threshold: float = RAIN_THRESHOLD_MM) -> pd.DataFrame:
    """`grid` is a continuous, hourly, issuance-specific weather timeline.

    Recent rain/duration calculations exclude target-hour precipitation. For
    durations, the last rainy run is truncated at 72 hours; 73 for hours-since
    means 'no rain in a complete previous 72-hour window'. Missing historical
    hours give NaN rather than being misinterpreted as dry weather.
    """
    result = pd.DataFrame(index=grid.index)
    rain = pd.to_numeric(grid.precipitation, errors='coerce')
    cloud = pd.to_numeric(grid.cloud_cover, errors='coerce')
    temperature = pd.to_numeric(grid.temperature_2m, errors='coerce')
    lagged = rain.shift(1)
    for hrs in (3, 6, 12):
        result[f'storm_rain_prev{hrs}h_mm'] = lagged.rolling(hrs, min_periods=hrs).sum().clip(lower=0)
    result['storm_peak_rain_prev12h_mm'] = lagged.rolling(12, min_periods=12).max().clip(lower=0)

    r = rain.to_numpy(dtype=float)
    wet = np.isfinite(r) & (r >= threshold)
    n = len(r)
    last = np.maximum.accumulate(np.where(wet, np.arange(n, dtype=int), -10_000))
    last_prior = np.r_[-10_000, last[:-1]]
    # At least the immediately preceding 72 hourly values must exist, even if
    # the most recent rain was already many hours ago.
    complete = lagged.notna().rolling(72, min_periods=72).sum().to_numpy() == 72
    elapsed = np.minimum(np.arange(n) - last_prior, 73).astype(float)
    elapsed[~complete] = np.nan
    result['storm_hours_since_rain'] = elapsed

    streak = np.zeros(n, dtype=int)
    for idx in range(n):
        if wet[idx]:
            streak[idx] = min(72, (streak[idx - 1] if idx else 0) + 1)
    safe_idx = np.clip(last_prior, 0, n - 1)
    duration = np.where((last_prior >= 0) & (elapsed <= 72), streak[safe_idx], 0).astype(float)
    duration[~complete] = np.nan
    result['storm_last_event_duration_h'] = duration

    # Rain-ending flag is only defined when both current and preceding hours
    # are known. This uses forecast current-hour rain, not realized current rain.
    ending = ((rain < threshold) & (lagged >= threshold)).astype(float)
    ending.loc[rain.isna() | lagged.isna()] = np.nan
    result['storm_rain_ended_this_hour'] = ending
    result['storm_cloud_change_3h'] = cloud - cloud.shift(3)
    result['storm_temperature_change_3h'] = temperature - temperature.shift(3)
    return result


def read_weather_sources(history_csv: str, archive_csv: str):
    # Uses the same validated parsers as the existing as-of pipeline.
    history = core.load_weather(history_csv)
    archive = asof.load_archive_csv(archive_csv)
    if history.empty or archive.empty:
        raise ValueError('Empty historical or archived weather source')
    return history, archive


def enrich_examples(examples: pd.DataFrame, history: pd.DataFrame,
                    archive: pd.DataFrame, *, progress_every: int = 1000) -> pd.DataFrame:
    """Compute one row of extra features per labeled as-of example.

    Requires full-hourly archived weather, NOT just store operating hours. Does
    not read target labels except to retain existing exact keys. Group-level
    processing avoids global sort/rolls that would mix forecast issuances.
    """
    if examples.duplicated(KEYS).any():
        raise ValueError('Duplicate example keys')
    if archive.duplicated(KEYS).any():
        raise ValueError('Duplicate archived weather keys')
    history_by_site = {
        site: part.sort_values('timestamp').set_index('timestamp')
        for site, part in history.groupby('location_id', sort=False)
    }
    by_issue = archive.groupby(['issued_at', 'location_id'], sort=False)
    outputs = []
    grouped = examples.groupby(['issued_at', 'location_id'], sort=False)
    for count, ((issued, site), targets) in enumerate(grouped, start=1):
        if site not in history_by_site:
            raise ValueError(f'No historical weather for location {site!r}')
        try:
            predicted = by_issue.get_group((issued, site))
        except KeyError as exc:
            raise ValueError(f'Archived weather missing issuance {issued}, store {site}') from exc
        # Never use reconstructed actual rain AFTER a forecast was issued.
        h = history_by_site[site]
        begin = issued - pd.Timedelta(hours=96)
        observed = h.loc[(h.index >= begin) & (h.index <= issued),
                         ['precipitation', 'temperature_2m', 'cloud_cover']]
        last_target = targets.timestamp.max()
        future = predicted[(predicted.timestamp > issued) &
                           (predicted.timestamp <= last_target)].copy()
        if future.empty:
            raise ValueError(f'No as-issued target weather for {site} at {issued}')
        future = future.set_index('timestamp')[['precipitation', 'temperature_2m', 'cloud_cover']]
        weather = pd.concat([observed, future]).sort_index()
        if weather.index.has_duplicates:
            raise ValueError(f'Overlapping as-of weather at {issued}, store {site}')
        timeline = pd.date_range(begin, last_target, freq='h', tz='UTC')
        weather = weather.reindex(timeline)
        # Fail early if the target-hour archived forecast is missing, rather
        # than implicitly imputing unknown target weather.
        idx = timeline.get_indexer(targets.timestamp)
        if (idx < 0).any():
            raise RuntimeError('Target timestamps unexpectedly outside hourly timeline')
        rain_at_target = weather.precipitation.to_numpy()[idx]
        if (not np.isfinite(rain_at_target).all() or
                not np.isclose(rain_at_target,
                               pd.to_numeric(targets.precipitation).to_numpy(dtype=float),
                               atol=1e-4, rtol=1e-5).all()):
            raise ValueError(f'As-issued target rain does not match saved examples for {site}, {issued}; '
                             'check that weather CSVs correspond to the original as-of fit')
        computed = derive_hourly_features(weather)
        sub = targets[KEYS].copy()
        sub.loc[:, list(FEATURES)] = computed.iloc[idx][list(FEATURES)].to_numpy()
        outputs.append(sub)
        if progress_every and count % progress_every == 0:
            print(f'  Storm feature groups: {count:,}', flush=True)
    out = pd.concat(outputs, ignore_index=True)
    if out.duplicated(KEYS).any() or len(out) != len(examples):
        raise RuntimeError('Storm feature output did not preserve example keys')
    return out.sort_values(KEYS).reset_index(drop=True)


def make_parser():
    parser = argparse.ArgumentParser(description='Build offline as-of storm features from full-hourly archived forecasts')
    parser.add_argument('--examples', required=True)
    parser.add_argument('--historical-weather-csv', required=True)
    parser.add_argument('--archive-csv', required=True)
    parser.add_argument('--output', default='storm_features_v1/storm_features.csv')
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    sources = {s: str(Path(getattr(args, s)).resolve())
               for s in ('examples', 'historical_weather_csv', 'archive_csv')}
    examples = backtest.load_examples(args.examples)
    print(f'Loaded {len(examples):,} as-of labeled examples', flush=True)
    history, archive = read_weather_sources(args.historical_weather_csv, args.archive_csv)
    print(f'Loaded {len(history):,} historical and {len(archive):,} archived weather hours', flush=True)
    extras = enrich_examples(examples, history, archive)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + '.tmp')
    extras.to_csv(temp, index=False)
    temp.replace(output)
    meta = {'created_utc': datetime.now(timezone.utc).isoformat(), 'version': 1,
            'inputs': sources, 'count': len(extras), 'feature_columns': list(FEATURES),
            'rain_threshold_mm_per_hour': RAIN_THRESHOLD_MM,
            'notes': ['Past weather uses reconstructed historical weather at/before issuance only.',
                      'Future weather uses that issuance archived forecast only, never actual future rain.',
                      'No history in a complete preceding 72-hour window => missing storm recency/duration.',
                      'These experimental features are not integrated into production forecast code.']}
    output.with_suffix('.meta.json').write_text(json.dumps(meta, indent=2))
    print(f'Wrote {len(extras):,} as-of storm feature rows to {output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
