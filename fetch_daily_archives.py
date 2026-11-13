"""Resume daily Open-Meteo Single Runs collection using a locations CSV; no traffic data required.

Requires authorization for the relevant API usage. The local request cap is deliberately
conservative, but cannot know API usage from other devices or the provider's weighted billing.

Example:
  python fetch_daily_archives.py --locations washcast_weather_grid_mapping.csv \
    --start-date 2026-06-10 --end-date 2026-10-20 \
    --output weather_data/daily_partial --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

import asof
import washcast as core


KEYS = ['issued_at', 'location_id', 'timestamp']
ARCHIVE_COLUMNS = ['issued_at'] + core.WEATHER_COLS


def read_locations(path: str) -> pd.DataFrame:
    x = pd.read_csv(path, dtype={'location_id': str})
    lat = 'api_grid_latitude' if 'api_grid_latitude' in x else 'latitude'
    lon = 'api_grid_longitude' if 'api_grid_longitude' in x else 'longitude'
    if not {'location_id',lat,lon}.issubset(x.columns):
        raise ValueError('Locations CSV requires location_id plus api_grid_latitude/api_grid_longitude or latitude/longitude')
    x = x[['location_id',lat,lon]].rename(columns={lat:'latitude',lon:'longitude'}).copy()
    x['location_id'] = x['location_id'].astype(str).str.strip()
    x['latitude'] = pd.to_numeric(x.latitude, errors='coerce')
    x['longitude'] = pd.to_numeric(x.longitude, errors='coerce')
    if x.location_id.eq('').any() or x.location_id.duplicated().any():
        raise ValueError('Locations must have nonempty, unique IDs')
    if x[['latitude','longitude']].isna().any().any() or (x.latitude.abs()>90).any() or (x.longitude.abs()>180).any():
        raise ValueError('Invalid coordinate in locations file')
    return x.sort_values('location_id').reset_index(drop=True)


def api_params(latitude: float, longitude: float, issued_at: pd.Timestamp) -> dict:
    run = asof.safely_available_run(issued_at)
    if run.date() < date.fromisoformat(asof.DATA_START):
        raise ValueError(f'Single Runs API unavailable before {asof.DATA_START}')
    params = {
        'latitude': round(float(latitude),5), 'longitude': round(float(longitude),5),
        'run': run.strftime('%Y-%m-%dT%H:%M'), 'models': asof.MODEL_ID,
        'hourly': asof.HOURS, 'forecast_days':10, 'timezone':'UTC',
        'timeformat':'unixtime',
    }
    if os.getenv('OPEN_METEO_API_KEY'):
        params['apikey'] = os.environ['OPEN_METEO_API_KEY']
    return params


def cached_request_path(cache_dir: str, params: dict) -> Path:
    url = os.getenv('OPEN_METEO_SINGLE_RUN_URL', asof.ARCHIVE_URL)
    hashed = hashlib.sha256((url+json.dumps(params,sort_keys=True)).encode()).hexdigest()
    return Path(cache_dir)/f'{hashed}.json'


def issued_timestamp(d: date, tz: str, hour: int) -> pd.Timestamp:
    return asof.issue_time(d,tz,hour).tz_convert('UTC')


def completed_pairs(archive: pd.DataFrame, horizon: int, tz: str, hour: int) -> set[tuple[pd.Timestamp,str]]:
    """Only skip a pair when every requested target day has archived operating-hour weather."""
    if archive.empty:
        return set()
    completed = set()
    for (issued,loc),group in archive.groupby(['issued_at','location_id'], sort=False):
        local_day = issued.tz_convert(tz).date()
        # If older archive issue hour differs, do not consider it a match.
        if issued_timestamp(local_day,tz,hour) != issued:
            continue
        target = core.future_hours(local_day+timedelta(days=1),horizon,tz,8,19,9)
        g = group.set_index('timestamp')
        if set(target).issubset(set(g.index)) and not g.loc[list(target),core.WEATHER_COLS[2:]].isna().any().any():
            completed.add((issued,str(loc)))
    return completed


def load_existing(path: Path) -> pd.DataFrame:
    if path.exists():
        return asof.load_archive_csv(str(path))
    return pd.DataFrame(columns=ARCHIVE_COLUMNS)


def load_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def record_request(ledger_path: Path, day: str, origin: str, site: str):
    """Count *attempted* outbound requests before sending, even if the API errors."""
    with ledger_path.open('a',encoding='utf-8') as f:
        f.write(json.dumps({'local_date':day,'when_utc':datetime.now(timezone.utc).isoformat(),
                            'origin':origin,'location_id':site})+'\n')
        f.flush()
        os.fsync(f.fileno())


def save_archive(path: Path, old: pd.DataFrame, chunks: list[pd.DataFrame]):
    if not chunks:
        return
    new = pd.concat([old]+chunks, ignore_index=True)
    new = new.sort_values(KEYS).drop_duplicates(KEYS,keep='last')
    tmp = path.with_suffix('.tmp.csv')
    new[ARCHIVE_COLUMNS].to_csv(tmp,index=False)
    asof.load_archive_csv(str(tmp))  # verify parse and key uniqueness before replacement
    tmp.replace(path)


def run(locations: str, start_date: str, end_date: str, output: str, cache_dir: str,
        max_new_requests: int=100, daily_request_cap: int=250,
        timezone_name: str='America/Chicago', issue_hour: int=18,
        horizon: int=2, dry_run: bool=False, sundays_only: bool=False) -> dict:
    if max_new_requests < 1 or daily_request_cap < 1:
        raise ValueError('Request limits must be positive')
    if horizon not in (1,2,7):
        raise ValueError('Supported forecast horizon: 1, 2, or 7 days')
    if not 0 <= issue_hour < 24:
        raise ValueError('Issue hour must be between 0 and 23')
    first,last=date.fromisoformat(start_date),date.fromisoformat(end_date)
    if first > last:
        raise ValueError('Start date must not be later than end date')
    sites=read_locations(locations)
    destination=Path(output)
    archive_path=destination/'archived_forecasts.csv'
    old=load_existing(archive_path)
    known=set(old.location_id.unique())
    unknown=known-set(sites.location_id)
    if unknown:
        raise ValueError(f'Archive has unknown locations not present in mapping: {sorted(unknown)}')
    complete=completed_pairs(old,horizon,timezone_name,issue_hour)
    missing=[]
    seen_keys=set()
    cache_hits=0
    for day in (first+timedelta(days=i) for i in range((last-first).days+1)):
        if sundays_only and day.weekday() != 6:
            continue
        issue=issued_timestamp(day,timezone_name,issue_hour)
        for site in sites.itertuples(index=False):
            if (issue,site.location_id) in complete:
                continue
            params=api_params(site.latitude,site.longitude,issue)
            filename=cached_request_path(cache_dir,params)
            key=str(filename)
            if key not in seen_keys:
                seen_keys.add(key)
                if filename.exists():
                    cache_hits+=1
            missing.append((day,issue,site,filename))
    uncached=len([k for k in seen_keys if not Path(k).exists()])
    print(f'Mapped locations: {len(sites)}; unique grid coordinates: {len(sites[["latitude","longitude"]].drop_duplicates())}')
    print(f'Existing archived issuance/site combinations: {len(complete)}')
    print(f'Missing issuance/site combinations in requested range: {len(missing)}')
    print(f'Approximate unique new HTTP requests needed (before retries): {uncached}')
    print(f'Existing matching cache responses: {cache_hits}')
    print('Existing combined archive is preserved and new runs are appended. No traffic data required.')
    if dry_run or not missing:
        return {'missing_pairs':len(missing),'unique_new_requests':uncached,'completed':0,'api_attempts':0}

    destination.mkdir(parents=True,exist_ok=True)
    ledger_path=destination/'weather_api_request_ledger.jsonl'
    local_date=datetime.now(ZoneInfo(timezone_name)).date().isoformat()
    attempted_today=sum(r.get('local_date')==local_date for r in load_ledger(ledger_path))
    used_here=0
    downloaded=[]
    processed=0
    stop_reason='all requested combinations obtained'
    session=requests.Session()  # No automatic retry of 429 responses.
    try:
        for day,issue,site,filename in missing:
            is_cached=filename.exists()
            if not is_cached:
                if used_here>=max_new_requests:
                    stop_reason=f'per-run limit of {max_new_requests} new requests reached'
                    break
                if attempted_today>=daily_request_cap:
                    stop_reason=f'local daily request cap of {daily_request_cap} reached'
                    break
                record_request(ledger_path,local_date,str(day),site.location_id)
                attempted_today+=1
                used_here+=1
            try:
                g=asof.get_archive_run(session,site.location_id,site.latitude,site.longitude,
                                       issue,cache_dir)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 'unknown'
                try:
                    payload = exc.response.json() if exc.response is not None else {}
                    detail = payload.get('reason') or str(payload)
                except (ValueError, AttributeError):
                    detail = exc.response.text if exc.response is not None else str(exc)
                # Show the *actual* API rejection reason; HTTP 400 alone is ambiguous.
                detail = str(detail).strip().replace('\n', ' ')[:700]
                model_run = asof.safely_available_run(issue).strftime('%Y-%m-%dT%H:%MZ')
                stop_reason = (f'Open-Meteo HTTP {status} for location {site.location_id}, '
                               f'model {asof.MODEL_ID}, run {model_run}: {detail}')
                print(f'\nSTOPPED: {stop_reason}', file=sys.stderr, flush=True)
                print('No more API requests will be sent. Existing downloaded data is preserved.',
                      file=sys.stderr, flush=True)
                break
            # Missing hours must never be silently substituted with guessed values.
            future=core.future_hours(day+timedelta(days=1),horizon,timezone_name,8,19,9)
            forecast=g.set_index('timestamp')
            if not set(future).issubset(set(forecast.index)):
                raise ValueError(f'Incomplete target weather for {site.location_id} issued {day}')
            if forecast.loc[list(future),core.WEATHER_COLS[2:]].isna().any().any():
                raise ValueError(f'Missing target weather fields for {site.location_id} issued {day}')
            downloaded.append(g[ARCHIVE_COLUMNS])
            processed+=1
            if processed % 25 == 0:
                print(f'Recovered {processed} new location/origin combinations, {used_here} actual requests',flush=True)
    finally:
        save_archive(archive_path,old,downloaded)
    print(f'Saved {len(downloaded)} additional location/origin runs to {archive_path}')
    print(f'New API requests attempted: {used_here}; locally tracked today: {attempted_today}')
    print(f'Stopped: {stop_reason}')
    return {'missing_pairs':len(missing),'completed':len(downloaded),
            'api_attempts':used_here,'stopped':stop_reason}


def main():
    p=argparse.ArgumentParser(description='Extend a daily Single Runs archive using location IDs and weather-grid coordinates')
    p.add_argument('--locations',required=True)
    p.add_argument('--start-date',required=True)
    p.add_argument('--end-date',required=True)
    p.add_argument('--output',default='weather_data/daily_partial')
    p.add_argument('--cache-dir',default='.weather_cache')
    p.add_argument('--max-new-requests',type=int,default=100,
                   help='Maximum new HTTP requests in this invocation (default 100)')
    p.add_argument('--daily-request-cap',type=int,default=250,
                   help='Locally tracked per-day request ceiling across repeated runs (default 250)')
    p.add_argument('--timezone',default='America/Chicago')
    p.add_argument('--issue-hour',type=int,default=18)
    p.add_argument('--forecast-days',type=int,default=2,choices=[1,2,7])
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--sundays-only',action='store_true',
                   help='Check/fetch Sunday 6 PM issuances only for weekly scheduling')
    args=p.parse_args()
    run(args.locations,args.start_date,args.end_date,args.output,args.cache_dir,
        args.max_new_requests,args.daily_request_cap,args.timezone,
        args.issue_hour,args.forecast_days,args.dry_run,args.sundays_only)

if __name__=='__main__':
    main()
