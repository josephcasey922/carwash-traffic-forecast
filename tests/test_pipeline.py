"""Offline regression tests for ingestion, causality, closures and end-to-end run."""
from __future__ import annotations
import json
from datetime import date
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import washcast


def test_hourly_aggregation_recorded_zero_and_missing_not_imputed(tmp_path):
    f=tmp_path/'traffic.csv'
    f.write_text('timestamp,location_id,latitude,longitude,carwashes\n'
                 '2026-11-01T09:10:00Z,A,33,-86,0\n'
                 '2026-11-01T10:15:00Z,A,33,-86,2\n'
                 '2026-11-01T10:45:00Z,A,33,-86,3\n'
                 '2026-11-01T12:00:00Z,A,33,-86,4\n')
    x=washcast.load_traffic(str(f),'America/Chicago')
    assert len(x)==3  # 11:00 is missing, not an added zero.
    assert x.carwashes.tolist()==[0,5,4]
    assert x.timestamp.dt.strftime('%H').tolist()==['09','10','12']


def test_naive_time_interpreted_as_local_and_can_reject_no_timezone(tmp_path):
    f=tmp_path/'traffic.csv'
    f.write_text('timestamp,location_id,latitude,longitude,carwashes\n'
                 '2026-11-02 09:00:00,A,33,-86,10\n')
    x=washcast.load_traffic(str(f),'America/Chicago')
    assert x.timestamp.iloc[0]==pd.Timestamp('2026-11-02T15:00:00Z')
    with pytest.raises(ValueError,match='timezone-naive'):
        washcast.load_traffic(str(f),None)


def test_rain_lags_are_past_only_even_when_future_rain_is_present():
    ts=pd.date_range('2026-10-01',periods=85,freq='h',tz='UTC')
    df=pd.DataFrame({'timestamp':ts,'location_id':'A','temperature_2m':20.,
                     'precipitation':0.,'cloud_cover':50.})
    df.loc[72,'precipitation']=10.
    feat=washcast.weather_features(df)
    assert feat.loc[72,'rain_prev24h']==0
    assert feat.loc[73,'rain_prev24h']==10
    assert feat.loc[72,'rain_prev72h']==0


def test_future_excludes_known_closure_and_dst_day_length():
    z='America/Chicago'
    # Spring-forward day has 23 true hourly instants when the schedule is 24 hours.
    assert len(washcast.future_hours(date(2026,3,8),1,z,0,24))==23
    # Fall-back day has 25 true hourly instants.
    assert len(washcast.future_hours(date(2026,11,1),1,z,0,24))==25


def test_offline_run_creates_outputs(tmp_path):
    demo=tmp_path/'demo'
    parser=washcast.create_parser()
    args=parser.parse_args(['demo','--output',str(demo),'--sites','2','--days','37',
                            '--future-days','4','--end-date','2026-10-30'])
    washcast.generate_demo(args)
    out=tmp_path/'results'
    run_args=parser.parse_args(['run','--traffic',str(demo/'traffic.csv'),
       '--weather-csv',str(demo/'weather.csv'),'--closures',str(demo/'closures.csv'),
       '--train-days','36','--validation-days','6','--forecast-days','3',
       '--forecast-start','2026-10-31','--models','baseline,poisson',
       '--output',str(out)])
    washcast.run(run_args)
    for file in ['model_comparison.csv','hourly_forecast.csv','daily_forecast.csv',
                 'validation_predictions.csv','data_quality.json','run_summary.json','best_model.joblib']:
        assert (out/file).exists(),file
    pred=pd.read_csv(out/'hourly_forecast.csv')
    assert not pred.empty
    assert pred.expected_washes.min()>=0
    assert pred.upper_80_approx.ge(pred.expected_washes).all()
    assert pred.lower_80_approx.le(pred.expected_washes).all()
    meta=json.loads((out/'run_summary.json').read_text())
    assert meta['site_count']==2
    assert 'optimistic' in meta['weather_observation_leakage_warning'].lower()


def test_scheduled_hours_are_8_to_19_except_sunday_9_to_19():
    tz='America/Chicago'
    monday=washcast.future_hours(date(2026,11,2),1,tz,8,19,9)
    sunday=washcast.future_hours(date(2026,11,8),1,tz,8,19,9)
    monday_local=monday.tz_convert(tz)
    sunday_local=sunday.tz_convert(tz)
    assert list(monday_local.hour)==list(range(8,19))
    assert list(sunday_local.hour)==list(range(9,19))


def test_infer_interior_missing_washes_and_leave_edges_and_closures(tmp_path):
    path=tmp_path/'raw.csv'
    # Verify the user's company-export columns including alphanumeric location ID.
    path.write_text('hour,location_code_id,longitude,latitude,num_carwashes\n'
        '2026-11-02T10:00:00-06:00,AL001,-86.28,33.61,5\n'
        '2026-11-02T12:00:00-06:00,AL001,-86.28,33.61,7\n'
        '2026-11-04T08:00:00-06:00,AL001,-86.28,33.61,1\n'
        '2026-11-04T10:00:00-06:00,AL001,-86.28,33.61,2\n')
    traffic=washcast.load_traffic(str(path),'America/Chicago')
    filled,report=washcast.fill_missing_open_hours(traffic,{('AL001',date(2026,11,3))},
                                                   'America/Chicago',8,19,9)
    hours=filled.timestamp.dt.tz_convert('America/Chicago').dt.strftime('%Y-%m-%d %H:%M').tolist()
    expected_first=[f'2026-11-02 {h:02d}:00' for h in range(10,19)]
    assert hours == expected_first+['2026-11-04 08:00','2026-11-04 09:00','2026-11-04 10:00']
    assert filled.carwashes.tolist()==[5,0,7]+[0]*6+[1,0,2]
    assert filled.inferred_zero.tolist()==[False,True,False]+[True]*6+[False,True,False]
    assert report['total_inferred_zero_hours']==8


def test_sunday_zero_fill_starts_at_nine():
    z='America/Chicago'
    ts=pd.to_datetime(['2026-11-07T18:00:00-06:00',
                       '2026-11-08T11:00:00-06:00',
                       '2026-11-08T13:00:00-06:00'],utc=True)
    traffic=pd.DataFrame({'location_id':['A']*3,'timestamp':ts,
                          'latitude':[33.]*3,'longitude':[-86.]*3,'carwashes':[5,6,7]})
    filled,report=washcast.fill_missing_open_hours(traffic,set(),z,8,19,9)
    loc=filled.timestamp.dt.tz_convert(z)
    sunday=filled[loc.dt.date==date(2026,11,8)]
    assert list(sunday.timestamp.dt.tz_convert(z).dt.hour)==[9,10,11,12,13]
    assert sunday.carwashes.tolist()==[0,0,6,0,7]
    assert report['total_inferred_zero_hours']==3
