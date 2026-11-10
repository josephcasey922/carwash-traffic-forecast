from datetime import date
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pytest

import asof
import washcast as core


def test_archive_run_precedes_issue_by_at_least_eight_hours():
    for day in [date(2026,1,15),date(2026,6,15)]:
        issue=asof.issue_time(day,'America/Chicago',18).tz_convert('UTC')
        run=asof.safely_available_run(issue)
        assert run.hour==12
        assert issue-run>=pd.Timedelta(hours=8)
        assert issue-run<=pd.Timedelta(hours=32)


def example_weather():
    ts=pd.date_range('2026-10-11T00:00:00Z',periods=240,freq='h')
    realized=pd.DataFrame({'timestamp':ts,'location_id':'A','temperature_2m':1.0,
                           'precipitation':0.0,'cloud_cover':5.0})
    issued=asof.issue_time(date(2026,10,12),'America/Chicago',18).tz_convert('UTC')
    # Archived weather is deliberately NOT the same as later observed weather.
    archive=realized[realized.timestamp>issued].copy()
    archive['issued_at']=issued
    archive['temperature_2m']=99.0
    archive['precipitation']=2.0
    return realized,archive,issued


def test_asof_uses_only_issued_future_weather():
    realized,archive,issued=example_weather()
    features=asof.combine_asof_weather(realized,archive,'A',issued,issued+pd.Timedelta(days=3))
    past=features[features.timestamp<=issued]
    future=features[features.timestamp>issued]
    assert not past.empty and not future.empty
    assert (past.temperature_2m==1).all()
    assert (future.temperature_2m==99).all()
    assert (future.precipitation==2).all()
    # Forecast precipitation, not later observed zero rain, enters prior-24h features.
    assert future.rain_prev24h.iloc[-1]>0


def test_training_does_not_synthesize_missing_washes():
    realized,archive,issued=example_weather()
    d=date(2026,10,12)
    target_hours=core.future_hours(d.replace(day=13),1,'America/Chicago',7,19)
    traffic=pd.DataFrame({'location_id':'A','timestamp':target_hours[:2],
                          'latitude':33.1,'longitude':-86.7,'carwashes':[0,17]})
    rows=asof.make_rows(traffic,realized,archive,[d],'America/Chicago',18,1,7,19,set(),True)
    assert len(rows)==2
    assert rows.carwashes.tolist()==[0,17]
    assert (rows.temperature_2m==99).all()


def test_duplicate_archive_issue_keys_rejected(tmp_path):
    path=tmp_path/'dup.csv'
    pd.DataFrame({'issued_at':['2026-10-12T23:00Z']*2,
                  'timestamp':['2026-10-13T10:00Z']*2,'location_id':['1']*2,
                  'temperature_2m':[10]*2,'precipitation':[0]*2,
                  'cloud_cover':[0]*2}).to_csv(path,index=False)
    with pytest.raises(ValueError,match='duplicate'):
        asof.load_archive_csv(str(path))


def test_training_and_validation_target_disjointness():
    # Even when a training-issued 7-day forecast reaches into validation dates,
    # those labels must be purged from model-fitting examples.
    times=pd.date_range('2026-10-01T00:00Z',periods=15,freq='D')
    earliest_validation=pd.Timestamp('2026-10-10T23:00Z')
    sample=pd.DataFrame({'issued_at':np.repeat(times[0],len(times)),
                         'timestamp':times})
    training=sample[(sample.issued_at<earliest_validation)&(sample.timestamp<earliest_validation)]
    assert training.timestamp.max()<earliest_validation


def test_archive_origin_filter_skips_missing_issuance_and_site():
    origins = [date(2026, 6, 9), date(2026, 6, 10), date(2026, 6, 11)]
    issues = [asof.issue_time(d, 'America/Chicago', 18).tz_convert('UTC') for d in origins]
    archive = pd.DataFrame({'issued_at': [issues[0], issues[0], issues[2]],
                            'location_id': ['AL002', 'AL003', 'AL002']})
    available, skipped = asof.filter_complete_archive_origins(
        archive, origins, {'AL002', 'AL003'}, 'America/Chicago', 18)
    assert available == [date(2026, 6, 9)]
    assert [x['issue_date'] for x in skipped] == ['2026-06-10', '2026-06-11']
    assert [x['sites_present'] for x in skipped] == [0, 1]


def test_archive_missing_origins_remain_errors_without_opt_in(tmp_path):
    issue = asof.issue_time(date(2026, 6, 9), 'America/Chicago', 18).tz_convert('UTC')
    path = tmp_path/'archive.csv'
    pd.DataFrame({'issued_at':[str(issue)], 'timestamp':['2026-06-10T10:00Z'],
                  'location_id':['AL002'], 'temperature_2m':[20],
                  'precipitation':[0], 'cloud_cover':[0]}).to_csv(path,index=False)
    with pytest.raises(ValueError, match='no forecasts'):
        asof.archive_for_origins(pd.DataFrame(),
            [date(2026, 6, 9),date(2026, 6, 10)],'America/Chicago',18,
            '.weather_cache', str(path))
    allowed = asof.archive_for_origins(pd.DataFrame(),
        [date(2026, 6, 9),date(2026, 6, 10)],'America/Chicago',18,
        '.weather_cache', str(path), allow_missing_origins=True)
    assert len(allowed) == 1
