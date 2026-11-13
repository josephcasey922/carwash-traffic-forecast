"""Leakage-safe, store-aware autoregressive features for WashCast.

This module turns the repeated, issuance-specific training-example labels into
ONE historical observation per store and UTC target hour. It never treats the
same wash count from two forecast issuances as two transactions.

At forecast issuance U, every lag and rolling statistic uses ONLY dates prior
to U's local calendar date. This intentionally discards the partially completed
issuance day, even when some of its transactions are already known. That rule
works for the 6 PM America/Chicago historical issuance schedule and remains
conservative if the issue time changes.

The source may have inferred zero hours. Complete, likely usable store-days are
used for rolling daily demand; missing store-days are NOT silently set to zero.
The source has no operational closure or outage truth, so data-quality checks
remain essential.

The design follows common pooled retail forecasting practice (store lags,
rolling means, current-vs-long-run change, store history and cold-start support).
This implementation is native pandas because we must preserve historical
forecast-vintage keys; it does not require the optional MLForecast package.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import backtest

KEYS = ['issued_at', 'location_id', 'timestamp']
WINDOWS = (7, 28, 56, 180)
FEATURES = (
    'rd_daily_mean_7d', 'rd_daily_mean_28d', 'rd_daily_mean_56d', 'rd_daily_mean_180d',
    'rd_daily_days_7d', 'rd_daily_days_28d', 'rd_daily_days_56d', 'rd_daily_days_180d',
    'rd_ratio_7d_56d', 'rd_ratio_28d_180d',
    'rd_hour_median_28d', 'rd_dow_hour_median_56d',
    'rd_lag_7d', 'rd_lag_14d', 'rd_lag_28d',
    'rd_store_age_days', 'rd_valid_days_total', 'rd_days_since_last_valid_day',
    'rd_network_dow_hour_median_56d', 'rd_inferred_fraction_28d',
)
SCHEMA_VERSION = 1


def unique_history(examples: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate target hours, rejecting contradictory copies.

    This checks target and inferred-zero consistency instead of silently picking
    one version. It should also reject target rows whose timestamps predate
    their own issuance or whose counts are invalid (backtest.load_examples).
    """
    required = set(KEYS + ['local_date', 'hour', 'dow', 'carwashes'])
    if not required.issubset(examples.columns):
        raise ValueError(f'Missing columns for history: {sorted(required - set(examples))}')
    if examples.duplicated(KEYS).any():
        raise ValueError('Duplicate forecast example key (issuance, store, target hour)')
    source = examples.copy()
    if 'inferred_zero' not in source.columns:
        source['inferred_zero'] = False
    source['inferred_zero'] = source['inferred_zero'].astype(bool)
    target_keys = ['location_id', 'timestamp']
    grp = source.groupby(target_keys, sort=False, observed=True)
    for col in ('carwashes', 'local_date', 'hour', 'dow', 'inferred_zero'):
        if grp[col].nunique(dropna=False).gt(1).any():
            raise ValueError(f'Inconsistent repeated observations for column {col!r}')
    history = (source.drop_duplicates(target_keys)[target_keys +
               ['local_date', 'hour', 'dow', 'carwashes', 'inferred_zero']].copy())
    history['day'] = pd.to_datetime(history.local_date).to_numpy(dtype='datetime64[D]')
    history['hour'] = pd.to_numeric(history.hour).astype(int)
    history['dow'] = pd.to_numeric(history.dow).astype(int)
    history = history.sort_values(['location_id','timestamp']).reset_index(drop=True)
    return history


def summarize_days(history: pd.DataFrame, *, max_inferred_per_day: int = 3) -> pd.DataFrame:
    """Daily totals with explicit completeness and label-quality flags.

    Sun 9:00-18:00 => 10 recorded hours; Mon-Sat 8:00-18:00 => 11.
    Inferred-zero hours are allowed up to the configurable cap. Entirely absent
    store-days are absent in this table, NOT imputed as zero-volume days.
    """
    if max_inferred_per_day < 0:
        raise ValueError('max_inferred_per_day must be nonnegative')
    group = history.groupby(['location_id','day'], sort=True, observed=True)
    daily = group.agg(daily_washes=('carwashes','sum'),
                      n_hours=('hour','nunique'),
                      n_inferred=('inferred_zero','sum'),
                      dow=('dow','first')).reset_index()
    daily['expected_hours'] = np.where(daily.dow.eq(6), 10, 11)
    daily['valid'] = ((daily.n_hours == daily.expected_hours)
                      & (daily.n_inferred <= max_inferred_per_day))
    return daily


def _prepare_arrays(history: pd.DataFrame, daily: pd.DataFrame) -> dict:
    stores = {}
    validity = daily[['location_id','day','valid']]
    hist = history.merge(validity, on=['location_id','day'], how='left', validate='many_to_one')
    for site, h in hist.groupby('location_id', sort=False, observed=True):
        d = daily[daily.location_id.eq(site)].sort_values('day')
        days = d.day.to_numpy(dtype='datetime64[D]').astype('int64')
        store_hours = h.sort_values('day')
        hd = store_hours.day.to_numpy(dtype='datetime64[D]').astype('int64')
        hh = store_hours.hour.to_numpy(dtype=int)
        yy = store_hours.carwashes.to_numpy(dtype=float)
        good = store_hours.valid.to_numpy(dtype=bool)
        stores[site] = {
            'days': days, 'daily': d.daily_washes.to_numpy(dtype=float),
            'valid': d.valid.to_numpy(dtype=bool),
            'hours_reported': d.n_hours.to_numpy(dtype=int),
            'hours_inferred': d.n_inferred.to_numpy(dtype=int),
            'hour_days': hd, 'hour_hours': hh, 'hour_dow': store_hours.dow.to_numpy(dtype=int),
            'hour_values': yy, 'hour_good': good,
            'lag_lookup': {(int(day),int(hour)): float(value) for day,hour,value in zip(hd,hh,yy)},
        }
    # All eligible past store hours for the pooled weekday/hour forecast prior.
    valid_hist = hist.loc[hist.valid].copy()
    network = {
        'day': valid_hist.day.to_numpy(dtype='datetime64[D]').astype('int64'),
        'dow': valid_hist.dow.to_numpy(dtype=int),
        'hour': valid_hist.hour.to_numpy(dtype=int),
        'value': valid_hist.carwashes.to_numpy(dtype=float),
    }
    return stores, network


def derive_features(examples: pd.DataFrame, *, timezone_name: str = 'America/Chicago',
                    max_inferred_per_day: int = 3, progress_every: int = 50) -> tuple[pd.DataFrame, dict]:
    """Compute all historical features efficiently, without future labels.

    The day cutoff is derived from each row's **issuance**, not target date.
    Store arrays and lag lookups are constructed once; no row-wise fits.
    """
    if progress_every < 1:
        raise ValueError('progress_every must be positive')
    ZoneInfo(timezone_name)
    if not isinstance(examples.issued_at.dtype, pd.DatetimeTZDtype):
        raise ValueError('issued_at must be a timezone-aware datetime')
    hist = unique_history(examples)
    daily = summarize_days(hist, max_inferred_per_day=max_inferred_per_day)
    stores, network = _prepare_arrays(hist, daily)
    # Arrays keep feature generation efficient even with >150k repeated labels.
    outputs = {f: np.full(len(examples), np.nan, dtype=float) for f in FEATURES}
    target_day = pd.to_datetime(examples.local_date).to_numpy(dtype='datetime64[D]').astype('int64')
    target_hour = examples.hour.to_numpy(dtype=int)
    target_dow = examples.dow.to_numpy(dtype=int)
    keys = examples.groupby(['issued_at','location_id'], sort=True, observed=True).indices
    issues = sorted(examples.issued_at.unique())
    groups_by_issue = {}
    for (issue, site), indices in keys.items():
        groups_by_issue.setdefault(pd.Timestamp(issue), []).append((site, indices))

    for i, issue in enumerate(issues, 1):
        issue = pd.Timestamp(issue)
        cutoff = int(np.datetime64(issue.tz_convert(timezone_name).date(),'D').astype('int64'))
        # A network-level profile is useful at launch, before a store has data.
        recent_net = ((network['day'] >= cutoff - 56) & (network['day'] < cutoff))
        # At most 7 * 11 pooled weekday/hour medians, shared across stores.
        net_dow, net_h, net_y = (network[k][recent_net] for k in ('dow','hour','value'))
        net_lookup = {}
        for dw in range(7):
            for hr in range(8,19):
                match = (net_dow == dw) & (net_h == hr)
                if match.any():
                    net_lookup[(dw,hr)] = float(np.median(net_y[match]))
        for site, indices in groups_by_issue.get(issue,[]):
            arr = stores[site]
            days = arr['days']; valid = arr['valid']; values = arr['daily']
            valid_past = (days < cutoff) & valid
            for n in WINDOWS:
                mask = valid_past & (days >= cutoff - n)
                nvalid = int(mask.sum())
                outputs[f'rd_daily_mean_{n}d'][indices] = float(values[mask].mean()) if nvalid else np.nan
                outputs[f'rd_daily_days_{n}d'][indices] = nvalid
            recent, long = outputs['rd_daily_mean_7d'][indices[0]], outputs['rd_daily_mean_56d'][indices[0]]
            ratio = recent / long if np.isfinite(recent) and np.isfinite(long) and long > 0 else np.nan
            outputs['rd_ratio_7d_56d'][indices] = ratio
            recent, long = outputs['rd_daily_mean_28d'][indices[0]], outputs['rd_daily_mean_180d'][indices[0]]
            ratio = recent / long if np.isfinite(recent) and np.isfinite(long) and long > 0 else np.nan
            outputs['rd_ratio_28d_180d'][indices] = ratio
            ndays = int(valid_past.sum())
            outputs['rd_valid_days_total'][indices] = ndays
            outputs['rd_store_age_days'][indices] = (cutoff - int(days[valid_past][0])) if ndays else 0.
            outputs['rd_days_since_last_valid_day'][indices] = (cutoff - int(days[valid_past][-1])) if ndays else np.nan
            seen = (days < cutoff) & (days >= cutoff-28)
            total_hours = int(arr['hours_reported'][seen].sum())
            outputs['rd_inferred_fraction_28d'][indices] = (float(arr['hours_inferred'][seen].sum()/total_hours)
                                                         if total_hours else np.nan)
            hd = arr['hour_days']; hh = arr['hour_hours']; hgood = arr['hour_good']; hy = arr['hour_values']; hwd = arr['hour_dow']
            valid28 = hgood & (hd >= cutoff-28) & (hd < cutoff)
            valid56 = hgood & (hd >= cutoff-56) & (hd < cutoff)
            local_days = target_day[indices]; local_hours = target_hour[indices]; local_dow = target_dow[indices]
            for hour in np.unique(local_hours):
                matched = valid28 & (hh == hour)
                outputs['rd_hour_median_28d'][indices[local_hours == hour]] = (float(np.median(hy[matched])) if matched.any() else np.nan)
            for day, hour in set(zip(local_dow, local_hours)):
                ix = indices[(local_dow == day) & (local_hours == hour)]
                matched = valid56 & (hwd == day) & (hh == hour)
                outputs['rd_dow_hour_median_56d'][ix] = float(np.median(hy[matched])) if matched.any() else np.nan
                outputs['rd_network_dow_hour_median_56d'][ix] = net_lookup.get((day,hour),np.nan)
            lookup = arr['lag_lookup']
            for lag in (7,14,28):
                outputs[f'rd_lag_{lag}d'][indices] = [
                    lookup.get((int(day-lag), int(hour)), np.nan) if day-lag < cutoff else np.nan
                    for day,hour in zip(local_days,local_hours)]
        if i % progress_every == 0 or i == len(issues):
            print(f'Recent demand features: {i}/{len(issues)} forecast issuances',flush=True)
    result = examples[KEYS].copy().reset_index(drop=True)
    for name in FEATURES:
        result[name] = outputs[name]
    if result.duplicated(KEYS).any() or len(result) != len(examples):
        raise RuntimeError('Recent-demand feature rows lost or duplicated')
    info = {
        'schema_version': SCHEMA_VERSION, 'forecast_examples': int(len(examples)),
        'unique_store_hours': int(len(hist)), 'unique_store_days': int(len(daily)),
        'eligible_store_days': int(daily.valid.sum()),
        'excluded_store_days_for_daily_statistics': int((~daily.valid).sum()),
        'unique_inferred_zero_hours': int(hist.inferred_zero.sum()),
        'issuances': int(len(issues)), 'time_zone': timezone_name,
        'max_inferred_per_day': int(max_inferred_per_day),
        'note': 'Features use distinct historical store-hours and only complete local calendar days before issuance; no source traffic is changed.'
    }
    return result, info


def make_parser():
    p = argparse.ArgumentParser(description='Offline leakage-safe per-store lag and recent-demand features')
    p.add_argument('--examples', required=True)
    p.add_argument('--timezone', default='America/Chicago')
    p.add_argument('--max-inferred-per-day', type=int, default=3)
    p.add_argument('--output', default='recent_demand_v1/recent_demand_features.csv')
    p.add_argument('--progress-every', type=int, default=50)
    return p


def main(argv=None):
    a = make_parser().parse_args(argv)
    examples = backtest.load_examples(a.examples)
    features, info = derive_features(examples, timezone_name=a.timezone,
                                     max_inferred_per_day=a.max_inferred_per_day,
                                     progress_every=a.progress_every)
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(path,index=False)
    info['created_at_utc'] = datetime.now(timezone.utc).isoformat()
    info['source_examples'] = str(Path(a.examples).resolve())
    (path.parent / 'recent_demand_info.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info, indent=2))
    print(f'Wrote {path.resolve()}',flush=True)


if __name__ == '__main__':
    main()
