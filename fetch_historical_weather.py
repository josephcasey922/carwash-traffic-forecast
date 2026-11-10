"""Incrementally backfill WashCast's *stitched historical-forecast* hourly weather.

This intentionally uses the SAME API as the existing historical_weather.csv:
https://historical-forecast-api.open-meteo.com/v1/forecast.  It is not ERA5,
station observations, or the Single Runs vintage forecast archive.

Compatible with WashCast's existing weather_data/daily_partial/historical_weather.csv.
The archived_forecasts.csv file is never opened for writing.

Example (PowerShell):
 .\\.venv\\Scripts\\python.exe fetch_historical_weather.py `
   --locations washcast_weather_grid_mapping.csv `
   --start-date 2025-02-01 --end-date 2026-11-08 `
   --output weather_data/daily_partial --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

COLUMNS = ['timestamp', 'location_id', 'temperature_2m', 'precipitation', 'cloud_cover']
VARIABLES = COLUMNS[2:]
HISTORY_URL = 'https://historical-forecast-api.open-meteo.com/v1/forecast'


def read_locations(filename: str) -> dict[tuple[float, float], list[str]]:
    x = pd.read_csv(filename, dtype={'location_id': str})
    lat_col = 'api_grid_latitude' if 'api_grid_latitude' in x else 'latitude'
    lon_col = 'api_grid_longitude' if 'api_grid_longitude' in x else 'longitude'
    if not {'location_id', lat_col, lon_col}.issubset(x.columns):
        raise ValueError('Missing location_id and latitude/longitude columns in locations CSV')
    x['location_id'] = x.location_id.astype(str).str.strip()
    if x.location_id.eq('').any() or x.location_id.duplicated().any():
        raise ValueError('Location IDs must be nonempty and unique')
    x[lat_col] = pd.to_numeric(x[lat_col], errors='raise')
    x[lon_col] = pd.to_numeric(x[lon_col], errors='raise')
    if x[[lat_col, lon_col]].isna().any().any() or x[lat_col].abs().gt(90).any() or x[lon_col].abs().gt(180).any():
        raise ValueError('Invalid locations coordinates')
    groups = defaultdict(list)
    for r in x.itertuples(index=False):
        # Use the official column names rather than the tuple position.
        loc = str(getattr(r, 'location_id'))
        lat, lon = float(getattr(r, lat_col)), float(getattr(r, lon_col))
        groups[(round(lat, 5), round(lon, 5))].append(loc)
    return dict(sorted(groups.items()))


def load_existing(filename: Path) -> pd.DataFrame:
    if not filename.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(filename, dtype={'location_id': str})
    if not set(COLUMNS).issubset(df.columns):
        raise ValueError(f'{filename} is missing expected columns: {set(COLUMNS) - set(df.columns)}')
    df = df[COLUMNS].copy()
    df['timestamp'] = pd.to_datetime(df.timestamp, utc=True, errors='raise')
    if df.timestamp.isna().any() or df.location_id.isna().any():
        raise ValueError('Existing weather data has missing keys')
    if (df.timestamp.dt.minute.ne(0) | df.timestamp.dt.second.ne(0)).any():
        raise ValueError('Existing weather contains non-hourly timestamps')
    if df.duplicated(['timestamp','location_id']).any():
        raise ValueError('Existing CSV contains duplicate (location_id,timestamp) keys; audit before backfill')
    for c in VARIABLES:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def complete_days(df: pd.DataFrame) -> set[tuple[str, date]]:
    if df.empty:
        return set()
    valid = df.loc[df[VARIABLES].notna().all(axis=1), ['location_id','timestamp']].copy()
    valid['day'] = valid.timestamp.dt.date
    n = valid.groupby(['location_id','day']).timestamp.nunique()
    return {tuple(k) for k in n.index[n.eq(24)]}


def plan_downloads(groups: dict[tuple[float,float],list[str]], complete: set[tuple[str,date]],
                   first: date, last: date, chunk_days: int = 14):
    if chunk_days < 1 or chunk_days > 14:
        raise ValueError('chunk_days must be between 1 and 14 to avoid weighted API requests')
    tasks = []
    missing_site_days = 0
    for coords, locations in groups.items():
        batch = []
        for i in range((last-first).days+2):
            d = first + timedelta(days=i)
            missing = tuple(loc for loc in locations if (loc,d) not in complete) if d <= last else ()
            if missing:
                missing_site_days += len(missing)
            if missing and (not batch or (d - batch[-1][0]).days == 1) and len(batch)<chunk_days:
                batch.append((d,missing))
            else:
                if batch:
                    tasks.append((coords, batch[0][0],batch[-1][0], tuple(batch)))
                    batch=[]
                if missing:
                    batch=[(d,missing)]
    # Processing chronological first makes earliest missing periods visible in results.
    tasks.sort(key=lambda x:(x[1],x[0]))
    return tasks, missing_site_days


def params_for(coords, first: date, last: date):
    return {'latitude':coords[0], 'longitude':coords[1],
            'start_date':first.isoformat(), 'end_date':last.isoformat(),
            'hourly':','.join(VARIABLES), 'timezone':'UTC', 'timeformat':'unixtime',
            **({'apikey':os.environ['OPEN_METEO_API_KEY']} if os.getenv('OPEN_METEO_API_KEY') else {})}


def cache_path(cache_dir: Path, url: str, params: dict) -> Path:
    # Same cache key algorithm as the WashCast core's json_get function.
    digest = hashlib.sha256((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    return cache_dir/f'{digest}.json'


def fetch_json(session, url: str, params: dict, cache: Path):
    if cache.exists():
        payload = json.loads(cache.read_text(encoding='utf-8'))
    else:
        resp = session.get(url, params=params, timeout=90)
        if not 200 <= resp.status_code < 300:
            detail = resp.text.strip().replace('\r',' ').replace('\n',' ')[:300]
            try:
                body = resp.json()
                if isinstance(body,dict):
                    detail = str(body.get('reason',detail))[:300]
            except ValueError:
                pass
            raise RuntimeError(f'Open-Meteo HTTP {resp.status_code}: {detail}')
        try:
            payload = resp.json()
        except ValueError as exc:
            preview = resp.text.strip().replace('\r',' ').replace('\n',' ')[:300]
            raise RuntimeError(f'Non-JSON Open-Meteo HTTP {resp.status_code}; content type '
                               f'{resp.headers.get("Content-Type","unknown")}; preview {preview!r}') from exc
        if not isinstance(payload,dict) or payload.get('error'):
            raise RuntimeError(f'Open-Meteo error: {payload}')
    if not isinstance(payload,dict) or payload.get('error'):
        raise RuntimeError(f'Cached Open-Meteo error: {payload}')
    return payload


def extract_rows(payload: dict, requested_days, coords) -> list[pd.DataFrame]:
    hourly = payload.get('hourly')
    if not isinstance(hourly,dict) or 'time' not in hourly:
        raise ValueError(f'API returned no hourly weather for {coords}')
    vals = hourly['time']
    timestamps = pd.to_datetime(vals,unit='s',utc=True,errors='raise')
    n = len(timestamps)
    weather = pd.DataFrame({'timestamp':timestamps})
    for field in VARIABLES:
        values = hourly.get(field)
        if values is None or len(values) != n:
            raise ValueError(f'API did not return a complete {field} vector at {coords}')
        weather[field] = pd.to_numeric(pd.Series(values),errors='coerce')
    weather['day'] = weather.timestamp.dt.date
    out = []
    for day, missing_locations in requested_days:
        part = weather.loc[weather.day.eq(day),['timestamp']+VARIABLES].copy()
        expected = pd.date_range(start=pd.Timestamp(day).tz_localize('UTC'), periods=24, freq='h')
        if (len(part) != 24 or part.timestamp.duplicated().any()
                or not part.timestamp.isin(expected).all() or part[VARIABLES].isna().any().any()):
            raise ValueError(f'Incomplete 24-hour weather for {day} at {coords}; no rows saved for this request')
        for loc in missing_locations:
            site = part.copy()
            site.insert(1,'location_id',loc)
            out.append(site[COLUMNS])
    return out


def count_today(ledger: Path, local_day: str) -> int:
    if not ledger.exists():
        return 0
    with ledger.open(encoding='utf-8') as f:
        return sum(1 for line in f if line.strip() and json.loads(line).get('local_date') == local_day)


def log_request(ledger: Path, local_day: str, coords, first: date, last: date):
    ledger.parent.mkdir(parents=True,exist_ok=True)
    with ledger.open('a',encoding='utf-8') as f:
        f.write(json.dumps({'local_date':local_day,
                            'when_utc':datetime.now(timezone.utc).isoformat(),
                            'kind':'historical_forecast',
                            'coordinates':coords,'start_date':str(first),'end_date':str(last)})+'\n')
        f.flush()
        os.fsync(f.fileno())


def save_csv(filename:Path, existing:pd.DataFrame, new_frames:list[pd.DataFrame]) -> pd.DataFrame:
    if not new_frames:
        return existing
    combined = pd.concat(([existing] if not existing.empty else [])+new_frames,ignore_index=True)
    combined = combined.drop_duplicates(['timestamp','location_id'],keep='last')
    combined = combined.sort_values(['location_id','timestamp']).reset_index(drop=True)
    # The temp file lives beside the final file so replacement is atomic on Windows.
    filename.parent.mkdir(parents=True,exist_ok=True)
    tmp = filename.with_suffix('.tmp.csv')
    try:
        combined[COLUMNS].to_csv(tmp,index=False)
        check = load_existing(tmp)
        if len(check) != len(combined):
            raise ValueError('Saved CSV verification changed record count; original preserved')
        tmp.replace(filename)
    finally:
        if tmp.exists():
            tmp.unlink()
    return combined


def run(locations:str, start_date:str, end_date:str, output:str='weather_data/daily_partial',
        cache_dir:str='.weather_cache', dry_run:bool=False,
        max_new_requests:int=800, daily_request_cap:int=8000,
        chunk_days:int=14, request_pause:float=0.25,
        ledger_timezone:str='America/Chicago', checkpoint_every:int=10,
        session=None):
    first,last = date.fromisoformat(start_date),date.fromisoformat(end_date)
    if first > last:
        raise ValueError('start-date is later than end-date')
    if last >= datetime.now(timezone.utc).date():
        raise ValueError('Choose an end-date before today in UTC; recent days may not yet be available')
    if not 1 <= chunk_days <= 14:
        raise ValueError('chunk-days must be in 1..14')
    if max_new_requests < 1 or daily_request_cap < 1 or request_pause < 0 or checkpoint_every < 1:
        raise ValueError('Request limits/period must be positive (pause can be zero)')
    groups=read_locations(locations)
    out=Path(output)
    filename=out/'historical_weather.csv'
    existing=load_existing(filename)
    all_locs={loc for group in groups.values() for loc in group}
    unknown=set(existing.location_id.unique()) - all_locs
    if unknown:
        raise ValueError(f'Existing CSV contains unrecognized location IDs: {sorted(unknown)}')
    tasks, missing_days = plan_downloads(groups,complete_days(existing),first,last,chunk_days)
    url = os.environ.get('OPEN_METEO_HISTORY_URL',HISTORY_URL)
    cache=Path(cache_dir)
    matching_cache=sum(cache_path(cache,url,params_for(coords,a,b)).exists() for coords,a,b,_ in tasks)
    local_day=datetime.now(ZoneInfo(ledger_timezone)).date().isoformat()
    ledger=out/'weather_api_request_ledger.jsonl'  # Shared with Single Runs downloader.
    already_today=count_today(ledger,local_day)
    print(f'Location IDs: {len(all_locs)}; unique weather grid coordinates: {len(groups)}')
    print(f'Existing historical weather rows: {len(existing):,}')
    print(f'Missing complete 24-hour location/days: {missing_days:,}')
    print(f'14-day-or-smaller API requests needed: {len(tasks):,} ({matching_cache:,} already cached)')
    print(f'API requests already logged today (all WashCast weather downloaders): {already_today:,}')
    print(f'API: {url} (stitched historical forecasts, not station observations)')
    print(f'Will write only {filename}; archived_forecasts.csv is untouched')
    if dry_run or not tasks:
        return {'needed':len(tasks),'missing_location_days':missing_days,'new_requests':0}
    created_session = session is None
    if created_session:
        session = requests.Session()
    pending=[]
    completed=0
    new_requests=0
    last_sent=None
    reason='all available historical weather downloaded'
    failed=False
    try:
        for coords, a, b, requested_days in tasks:
            params=params_for(coords,a,b)
            cp=cache_path(cache,url,params)
            if not cp.exists():
                if new_requests>=max_new_requests:
                    reason=f'per-run limit {max_new_requests} reached'; break
                if already_today + new_requests >= daily_request_cap:
                    reason=f'local daily cap {daily_request_cap} reached'; break
                if last_sent is not None:
                    pause=request_pause - (time.monotonic()-last_sent)
                    if pause>0:time.sleep(pause)
                log_request(ledger,local_day,coords,a,b)
                new_requests+=1
                last_sent=time.monotonic()
            payload=fetch_json(session,url,params,cp)
            parsed = extract_rows(payload,requested_days,coords)
            # Cache only a verified, complete response, so malformed API data cannot
            # poison retries on future invocations.
            if not cp.exists():
                cp.parent.mkdir(parents=True,exist_ok=True)
                tmp_cache = cp.with_suffix('.tmp')
                tmp_cache.write_text(json.dumps(payload),encoding='utf-8')
                tmp_cache.replace(cp)
            pending.extend(parsed)
            completed+=1
            if completed % checkpoint_every==0:
                existing=save_csv(filename,existing,pending)
                pending=[]
                print(f'Completed {completed}/{len(tasks)} blocks; {new_requests} outbound requests; saved {len(existing):,} rows',flush=True)
    except (requests.RequestException, RuntimeError, ValueError, OSError) as exc:
        reason=f'{type(exc).__name__}: {exc}'
        failed=True
        print(f'STOPPED: {reason}',file=sys.stderr,flush=True)
    finally:
        existing=save_csv(filename,existing,pending)
        if created_session:
            session.close()
    remaining=len(tasks)-completed
    print(f'Saved historical weather rows: {len(existing):,} ({completed}/{len(tasks)} request blocks completed)')
    print(f'New HTTP requests attempted: {new_requests}; logged today: {already_today+new_requests}')
    print(f'Blocks remaining: {remaining}; stopped: {reason}')
    return {'needed':len(tasks),'completed':completed,'remaining':remaining,
            'new_requests':new_requests,'reason':reason,'failed':failed,'rows':len(existing)}


def main():
    p=argparse.ArgumentParser(description='Backfill WashCast historical hourly weather independently of traffic')
    p.add_argument('--locations',required=True)
    p.add_argument('--start-date',required=True)
    p.add_argument('--end-date',required=True)
    p.add_argument('--output',default='weather_data/daily_partial')
    p.add_argument('--cache-dir',default='.weather_cache')
    p.add_argument('--max-new-requests',type=int,default=800)
    p.add_argument('--daily-request-cap',type=int,default=8000)
    p.add_argument('--chunk-days',type=int,default=14)
    p.add_argument('--request-pause',type=float,default=0.25)
    p.add_argument('--checkpoint-every',type=int,default=10)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    result=run(a.locations,a.start_date,a.end_date,a.output,a.cache_dir,a.dry_run,
        a.max_new_requests,a.daily_request_cap,a.chunk_days,a.request_pause,
        checkpoint_every=a.checkpoint_every)
    if result.get('failed'):
        raise SystemExit(1)


if __name__=='__main__':
    main()
