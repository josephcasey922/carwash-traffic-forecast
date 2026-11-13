"""Label evaluated Sunday-issued 7-day WashCast forecasts for dashboard use.

No training, weather calls or traffic modifications. Converts the existing
7-day chronological benchmark output to a distinct model namespace so it
can be loaded alongside *daily-issued* benchmark predictions safely.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

MODEL_PREFIX = 'weekly_sun__'
KEYS = ['issued_at', 'location_id', 'timestamp', 'model']
NEEDED = set(KEYS + ['local_date', 'lead_day', 'carwashes', 'prediction', 'hour'])


def convert_weekly_predictions(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = NEEDED - set(raw.columns)
    if missing:
        raise ValueError(f'Not benchmark predictions; missing {sorted(missing)}')
    d = raw.copy()
    if d.empty:
        raise ValueError('Empty weekly backtest predictions')
    d['issued_at'] = pd.to_datetime(d['issued_at'], utc=True, errors='raise')
    d['timestamp'] = pd.to_datetime(d['timestamp'], utc=True, errors='raise')
    d['local_date'] = pd.to_datetime(d['local_date'], errors='raise').dt.normalize()
    d['lead_day'] = pd.to_numeric(d['lead_day'], errors='raise').astype('int16')
    issue_local = d['issued_at'].dt.tz_convert('America/Chicago')
    issue_dates = issue_local.dt.tz_localize(None).dt.normalize()
    actual_leads = (d['local_date']-issue_dates).dt.days
    if (issue_local.dt.dayofweek != 6).any():
        raise ValueError('A weekly forecast must have been issued on Sunday in America/Chicago')
    if (issue_local.dt.hour != 18).any() or (issue_local.dt.minute != 0).any():
        raise ValueError('The Sunday weekly forecast must be issued at 6 PM Chicago local time')
    if (~d['lead_day'].between(1, 7)).any() or not (d['lead_day'] == actual_leads).all():
        raise ValueError('Weekly target days must be Monday through the next Sunday (lead_day 1–7)')
    if (d['timestamp'] <= d['issued_at']).any():
        raise ValueError('Target timestamps are not after their forecast issuance')
    if (d['model'].astype(str).str.startswith(MODEL_PREFIX)).any():
        raise ValueError('Predictions are already labelled as weekly; do not convert twice')
    if d.duplicated(KEYS).any():
        raise ValueError('Duplicate Sunday issue/store/hour/model predictions')
    if d[['carwashes', 'prediction']].isna().any().any():
        raise ValueError('Weekly backtest contains missing actuals or predictions')
    d['source_model'] = d['model'].astype(str)
    d['model'] = MODEL_PREFIX + d['source_model']
    d['forecast_type'] = 'sunday_weekly'
    d['sunday_issue_date'] = issue_dates.dt.strftime('%Y-%m-%d')
    # A week has Mon-Sat 11 hours each plus Sunday 10 hours, 76 hours total.
    coverage = (d.groupby(['model', 'issued_at', 'location_id'], observed=True)
        .agg(target_hours=('timestamp','nunique'), target_days=('local_date','nunique'),
             first_day=('local_date','min'), last_day=('local_date','max'))
        .reset_index())
    coverage['expected_hours'] = 76
    coverage['complete_week'] = (coverage['target_hours'] == 76) & (coverage['target_days'] == 7)
    # Partial forecast weeks should not be described as full-week evaluations.
    return d, coverage


def main(argv=None):
    p = argparse.ArgumentParser(description='Convert Sunday 7-day backtest predictions for WashCast dashboard')
    p.add_argument('--predictions', required=True, help='The seven-day benchmark all_predictions.csv')
    p.add_argument('--output', default='sunday_weekly_oct_nov/weekly_predictions.csv')
    args = p.parse_args(argv)
    df, coverage = convert_weekly_predictions(pd.read_csv(args.predictions, dtype={'location_id':str}))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    coverage.to_csv(destination.parent/'weekly_coverage.csv', index=False)
    origins = df.issued_at.nunique()
    complete = int(coverage.complete_week.sum())
    print(f'Converted {len(df):,} weekly forecast hours from {origins} Sunday issuances')
    print(f'Complete store-weeks: {complete} / {len(coverage)}; incomplete weeks are flagged')
    print(f'Load this file in the dashboard: {destination}')


if __name__ == '__main__':
    main()
