import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import fetch_daily_archives as fa
import asof
import washcast as core


def _rows(origin: date, loc: str):
    issued=fa.issued_timestamp(origin,'America/Chicago',18)
    future=core.future_hours(origin+timedelta(days=1),2,'America/Chicago',8,19,9)
    return pd.DataFrame({'issued_at':issued,'timestamp':future,'location_id':loc,
                         'temperature_2m':14.,'precipitation':0.,'cloud_cover':20.})


def test_load_coordinate_mapping(tmp_path):
    path=tmp_path/'loc.csv'
    path.write_text('location_id,api_grid_latitude,api_grid_longitude\nAL002,33.56766,-86.542755\nAL003,33.56766,-86.3197\n')
    x=fa.read_locations(str(path))
    assert x.location_id.tolist()==['AL002','AL003']
    assert float(x.iloc[0].latitude)==33.56766


def test_dry_run_does_not_request_or_write(tmp_path,monkeypatch):
    loc=tmp_path/'loc.csv'
    loc.write_text('location_id,latitude,longitude\nAL002,33.5,-86.5\n')
    out=tmp_path/'out'
    monkeypatch.setattr(asof,'get_archive_run',lambda *a,**k: (_ for _ in ()).throw(AssertionError('API called')))
    result=fa.run(str(loc),'2026-06-10','2026-06-10',str(out),str(tmp_path/'cache'),dry_run=True)
    assert result['unique_new_requests']==1
    assert not out.exists()


def test_resume_preserves_existing_csv_and_caps_new_requests(tmp_path,monkeypatch):
    loc=tmp_path/'loc.csv'
    loc.write_text('location_id,api_grid_latitude,api_grid_longitude\nAL002,33.5,-86.5\nAL003,33.5,-86.5\nAL004,33.6,-86.4\n')
    out=tmp_path/'out'
    out.mkdir()
    initial=_rows(date(2026,6,9),'AL002')
    initial.to_csv(out/'archived_forecasts.csv',index=False)
    def fake_get(session,site,lat,lon,issue,cache):
        d=issue.tz_convert('America/Chicago').date()
        params=fa.api_params(lat,lon,issue)
        path=fa.cached_request_path(str(cache),params)
        path.parent.mkdir(exist_ok=True,parents=True)
        if not path.exists():
            path.write_text('{}')
        return _rows(d,site)
    monkeypatch.setattr(asof,'get_archive_run',fake_get)
    first=fa.run(str(loc),'2026-06-10','2026-06-10',str(out),str(tmp_path/'cache'),
                 max_new_requests=1,daily_request_cap=2)
    assert first['completed']==2  # two site IDs, one shared grid/cache call
    assert first['api_attempts']==1
    second=fa.run(str(loc),'2026-06-10','2026-06-10',str(out),str(tmp_path/'cache'),
                  max_new_requests=1,daily_request_cap=2)
    assert second['completed']==1
    assert second['api_attempts']==1
    third=fa.run(str(loc),'2026-06-10','2026-06-10',str(out),str(tmp_path/'cache'),
                 max_new_requests=1,daily_request_cap=2)
    assert third['completed']==0
    assert third['api_attempts']==0
    saved=asof.load_archive_csv(str(out/'archived_forecasts.csv'))
    assert set(saved.location_id)=={'AL002','AL003','AL004'}
    assert saved.duplicated(fa.KEYS).sum()==0
    assert len(saved)==4*len(core.future_hours(date(2026,6,10),2,'America/Chicago',8,19,9))
    ledger=fa.load_ledger(out/'weather_api_request_ledger.jsonl')
    assert len(ledger)==2


def _http_error(status: int, reason: str):
    import requests
    r=requests.Response()
    r.status_code=status
    r._content=json.dumps({'error':True,'reason':reason}).encode('utf-8')
    r.url='https://single-runs-api.open-meteo.com/v1/forecast'
    return requests.HTTPError(f'{status} error',response=r)


def test_missing_model_run_is_skipped_and_remembered(tmp_path,monkeypatch):
    loc=tmp_path/'loc.csv'
    loc.write_text('location_id,latitude,longitude\nAL002,33.5,-86.5\nAL003,33.5,-86.5\nAL004,33.6,-86.4\n')
    out=tmp_path/'out'
    cache=tmp_path/'cache'
    calls=[]
    def fake_get(session,site,lat,lon,issue,cachedir):
        d=issue.tz_convert('America/Chicago').date()
        calls.append((d,site))
        if d == date(2026,6,10):
            raise _http_error(400,'The requested model run is not available. Model: ecmwf_ifs, run: 2026-06-10T12:00Z')
        filename=fa.cached_request_path(str(cachedir),fa.api_params(lat,lon,issue))
        filename.parent.mkdir(parents=True,exist_ok=True)
        filename.write_text('{}')
        return _rows(d,site)
    monkeypatch.setattr(asof,'get_archive_run',fake_get)
    result=fa.run(str(loc),'2026-06-10','2026-06-11',str(out),str(cache),
                  max_new_requests=10,daily_request_cap=10)
    assert result['api_attempts']==3  # One missing run, two unique grid calls on June 11
    assert result['unavailable_pairs']==3
    assert result['unavailable_runs_added']==1
    assert result['completed']==3
    assert sum(d==date(2026,6,10) for d,site in calls)==1
    missing_file=out/'unavailable_forecast_runs.jsonl'
    assert missing_file.exists()
    skipped=fa.unavailable_runs(missing_file)
    assert skipped=={('ecmwf_ifs','2026-06-10T12:00Z')}
    saved=asof.load_archive_csv(str(out/'archived_forecasts.csv'))
    assert len(saved)==3*len(core.future_hours(date(2026,6,12),2,'America/Chicago',8,19,9))
    assert saved.issued_at.dt.tz_convert('America/Chicago').dt.date.nunique()==1
    # Subsequent runs must not call the known missing run, or completed pairs.
    calls.clear()
    next_result=fa.run(str(loc),'2026-06-10','2026-06-11',str(out),str(cache),
                       max_new_requests=10,daily_request_cap=10)
    assert next_result['api_attempts']==0
    assert not calls
    assert next_result['completed']==0
    assert next_result['missing_pairs']==0
    # An explicit retry remains possible later, but has to cost a request.
    retried=fa.run(str(loc),'2026-06-10','2026-06-10',str(out),str(cache),
                   max_new_requests=10,daily_request_cap=10,retry_unavailable_runs=True)
    assert retried['api_attempts']==1
    assert retried['unavailable_pairs']==3
    assert len(fa.load_ledger(missing_file))==1


def test_other_http_400_stops_instead_of_skipping(tmp_path,monkeypatch):
    loc=tmp_path/'loc.csv'
    loc.write_text('location_id,latitude,longitude\nAL002,33.5,-86.5\n')
    out=tmp_path/'out'
    monkeypatch.setattr(asof,'get_archive_run',lambda *a,**k:
                        (_ for _ in ()).throw(_http_error(400,'Invalid API key')))
    r=fa.run(str(loc),'2026-06-10','2026-06-11',str(out),str(tmp_path/'cache'),
             max_new_requests=10,daily_request_cap=10)
    assert r['api_attempts']==1
    assert r['completed']==0
    assert 'Invalid API key' in r['stopped']
    assert not (out/'unavailable_forecast_runs.jsonl').exists()
