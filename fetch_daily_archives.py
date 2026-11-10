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
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

import asof
import washcast as core


KEYS = ['issued_at', 'location_id', 'timestamp']
ARCHIVE_COLUMNS = ['issued_at'] + core.WEATHER_COLS


class UnexpectedWeatherResponse(RuntimeError):
    """The API returned a successful HTTP response without usable JSON."""


def validate_api_response(response: requests.Response, *args, **kwargs) -> requests.Response:
    """Catch HTML/empty responses before the core CSV downloader tries resp.json().

    Keep ordinary HTTP errors for the existing HTTPError handler, including the
    narrowly scoped HTTP 400 'run unavailable' rule.
    """
    if not 200 <= response.status_code < 300:
        return response
    try:
        payload = response.json()
    except ValueError as exc:
        content_type = response.headers.get('Content-Type', '(not supplied)')
        preview = response.text.strip().replace('\r', ' ').replace('\n', ' ')[:350]
        if not preview:
            preview = '(empty response)'
        raise UnexpectedWeatherResponse(
            f'HTTP {response.status_code}, Content-Type {content_type}; '
            f'expected JSON but got: {preview!r}'
        ) from exc
    if not isinstance(payload, dict):
        raise UnexpectedWeatherResponse(
            f'HTTP {response.status_code}; expected a JSON object, '
            f'got {type(payload).__name__}'
        )
    return response


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
    """Only skip a pair when both forecast days have all operating hours/weather."""
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


def model_run_key(issued_at: pd.Timestamp) -> tuple[str, str]:
    """An unavailable forecast run is independent of the site coordinates."""
    run_time = asof.safely_available_run(issued_at).strftime('%Y-%m-%dT%H:%MZ')
    return asof.MODEL_ID, run_time


def unavailable_runs(path: Path) -> set[tuple[str, str]]:
    return {(r['model'], r['run']) for r in load_ledger(path)
            if isinstance(r, dict) and 'model' in r and 'run' in r}


def record_unavailable(path: Path, model: str, run_time: str, reason: str) -> None:
    """Remember confirmed missing model runs so reruns don't waste API requests."""
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps({'model': model, 'run': run_time,
                            'recorded_utc': datetime.now(timezone.utc).isoformat(),
                            'reason': reason})+'\n')
        f.flush()
        os.fsync(f.fileno())


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
        horizon: int=2, dry_run: bool=False, retry_unavailable_runs: bool=False,
        request_pause: float=0.25) -> dict:
    if request_pause < 0:
        raise ValueError('Request pause must be nonnegative')
    if max_new_requests < 1 or daily_request_cap < 1:
        raise ValueError('Request limits must be positive')
    if horizon not in (1,2):
        raise ValueError('Supported forecast horizon: 1 or 2 days')
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
    unavailable_path=destination/'unavailable_forecast_runs.jsonl'
    skipped_runs=unavailable_runs(unavailable_path)
    missing=[]
    already_unavailable_pairs=0
    seen_keys=set()
    cache_hits=0
    for day in (first+timedelta(days=i) for i in range((last-first).days+1)):
        issue=issued_timestamp(day,timezone_name,issue_hour)
        for site in sites.itertuples(index=False):
            if (issue,site.location_id) in complete:
                continue
            if not retry_unavailable_runs and model_run_key(issue) in skipped_runs:
                already_unavailable_pairs += 1
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
    if already_unavailable_pairs:
        print(f'Known unavailable-run site/date combinations excluded: {already_unavailable_pairs} '
              '(not downloaded or silently filled)')
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
    newly_unavailable_pairs=0
    newly_unavailable_runs=set()
    stop_reason='all requested combinations obtained'
    session=requests.Session()  # No automatic retry of 429 responses.
    session.hooks['response'].append(validate_api_response)
    last_request_started = None
    try:
        for day,issue,site,filename in missing:
            model_run=model_run_key(issue)
            if model_run in newly_unavailable_runs:
                newly_unavailable_pairs += 1
                continue
            is_cached=filename.exists()
            if not is_cached:
                if used_here>=max_new_requests:
                    stop_reason=f'per-run limit of {max_new_requests} new requests reached'
                    break
                if attempted_today>=daily_request_cap:
                    stop_reason=f'local daily request cap of {daily_request_cap} reached'
                    break
                # Keep bulk backfills comfortably below the free API's per-minute cap.
                if last_request_started is not None and request_pause:
                    remaining = request_pause - (time.monotonic() - last_request_started)
                    if remaining > 0:
                        time.sleep(remaining)
                record_request(ledger_path,local_date,str(day),site.location_id)
                last_request_started = time.monotonic()
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
                model_name, run_time = model_run
                if status == 400 and 'requested model run is not available' in detail.casefold():
                    newly_unavailable_runs.add(model_run)
                    newly_unavailable_pairs += 1
                    if model_run not in skipped_runs:
                        record_unavailable(unavailable_path, model_name, run_time, detail)
                        skipped_runs.add(model_run)
                    print(f'Unavailable forecast run {model_name} {run_time}; '
                          'skipping all locations for this run. No substitute data invented.',
                          flush=True)
                    continue
                stop_reason = (f'Open-Meteo HTTP {status} for location {site.location_id}, '
                               f'model {model_name}, run {run_time}: {detail}')
                print(f'\nSTOPPED: {stop_reason}', file=sys.stderr, flush=True)
                print('No more API requests will be sent. Existing downloaded data is preserved.',
                      file=sys.stderr, flush=True)
                break
            except UnexpectedWeatherResponse as exc:
                stop_reason = (f'Unexpected non-JSON Open-Meteo response for '
                               f'{site.location_id} on {day}: {exc}')
                print(f'\nSTOPPED: {stop_reason}', file=sys.stderr, flush=True)
                print('No retries or further requests. Successful forecasts are saved.',
                      file=sys.stderr, flush=True)
                break
            except requests.RequestException as exc:
                stop_reason = (f'Open-Meteo connection/request error for '
                               f'{site.location_id} on {day}: {type(exc).__name__}: {exc}')
                print(f'\nSTOPPED: {stop_reason}', file=sys.stderr, flush=True)
                print('No retries or further requests. Successful forecasts are saved.',
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
    if stop_reason == 'all requested combinations obtained' and newly_unavailable_runs:
        stop_reason = (f'all available combinations retrieved; '
                       f'{len(newly_unavailable_runs)} model run(s) unavailable and left missing')
    print(f'Saved {len(downloaded)} additional location/origin runs to {archive_path}')
    if newly_unavailable_runs:
        print(f'Confirmed unavailable model runs this invocation: {len(newly_unavailable_runs)}; '
              f'site/date combinations left missing: {newly_unavailable_pairs}')
        print(f'Unavailable runs recorded in {unavailable_path}; '
              'use --retry-unavailable-runs to check them again later.')
    print(f'New API requests attempted: {used_here}; locally tracked today: {attempted_today}')
    print(f'Stopped: {stop_reason}')
    return {'missing_pairs':len(missing),'completed':len(downloaded),
            'api_attempts':used_here,'stopped':stop_reason,
            'unavailable_pairs':already_unavailable_pairs+newly_unavailable_pairs,
            'unavailable_runs_added':len(newly_unavailable_runs)}


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
    p.add_argument('--forecast-days',type=int,default=2,choices=[1,2])
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--retry-unavailable-runs',action='store_true',
                   help='Recheck previously unavailable model runs (costs API calls)')
    p.add_argument('--request-pause',type=float,default=0.25,
                   help='Minimum seconds between new requests (default 0.25)')
    args=p.parse_args()
    run(args.locations,args.start_date,args.end_date,args.output,args.cache_dir,
        args.max_new_requests,args.daily_request_cap,args.timezone,
        args.issue_hour,args.forecast_days,args.dry_run,args.retry_unavailable_runs,
        args.request_pause)

if __name__=='__main__':
    main()
