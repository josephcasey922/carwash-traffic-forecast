from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sunday_weekly_prepare import convert_weekly_predictions, MODEL_PREFIX
from sunday_weekly_forecast import sunday_issue, training_cutoff
from washcast_dashboard_core import (
    prepare_predictions, prepare_upcoming_weekly, matched_daily_weekly,
    sunday_weekly_rows, discover_prediction_csvs, metrics,
)


def sample_week(start='2026-11-01', model='lgbm_stable_recent'):
    sunday=pd.Timestamp(start)
    issue=(sunday+pd.Timedelta(hours=18)).tz_localize('America/Chicago')
    rows=[]
    for offset in range(1,8):
        current=(sunday+pd.Timedelta(days=offset))
        for hour in range(9 if current.weekday()==6 else 8,19):
            target=(current+pd.Timedelta(hours=hour)).tz_localize('America/Chicago')
            rows.append({'issued_at':issue.tz_convert('UTC'),'location_id':'AL003',
                         'timestamp':target.tz_convert('UTC'),
                         'local_date':current.strftime('%Y-%m-%d'),
                         'hour':hour,'lead_day':offset,'carwashes':hour+1,
                         'model':model,'prediction':float(hour)+.5})
    return pd.DataFrame(rows)


def test_convert_sunday_week_handles_dst_and_lead_seven():
    raw=sample_week()
    out,coverage=convert_weekly_predictions(raw)
    assert len(out)==76
    assert out.model.eq('weekly_sun__lgbm_stable_recent').all()
    assert out.lead_day.max()==7
    assert out.local_date.max()==pd.Timestamp('2026-11-08')
    assert out.issued_at.iloc[0]==pd.Timestamp('2026-11-02T00:00:00Z')
    assert coverage.complete_week.iloc[0]
    assert 'source_model' in out
    prepared=prepare_predictions(out)
    assert metrics(prepared)['n']==76


def test_reject_monday_issue_and_wrong_lead():
    raw=sample_week()
    broken=raw.copy()
    broken['issued_at']=pd.Timestamp('2026-11-03T00:00:00Z')
    with pytest.raises(ValueError,match='Sunday'):
        convert_weekly_predictions(broken)
    wrong=raw.copy()
    wrong.loc[0,'lead_day']=2
    with pytest.raises(ValueError,match='lead_day'):
        convert_weekly_predictions(wrong)


def test_reject_duplicates_and_partial_week_flagged():
    df=sample_week()
    partial,coverage=convert_weekly_predictions(df.iloc[:-2])
    assert not coverage.complete_week.iloc[0]
    dup=pd.concat([df,df.iloc[[0]]],ignore_index=True)
    with pytest.raises(ValueError,match='Duplicate'):
        convert_weekly_predictions(dup)


def test_match_weekly_with_daily_by_target_hour():
    weekly, _=convert_weekly_predictions(sample_week())
    prepared_weekly=prepare_predictions(weekly)
    daily=sample_week()
    # each day was issued the previous day at 6PM local
    day=(pd.to_datetime(daily.local_date))
    prior=day-pd.Timedelta(days=1)+pd.Timedelta(hours=18)
    daily['issued_at']=prior.dt.tz_localize('America/Chicago').dt.tz_convert('UTC')
    daily['lead_day']=1
    prepared_daily=prepare_predictions(daily)
    matched=matched_daily_weekly(prepared_daily,prepared_weekly)
    assert len(matched)==76
    assert matched.lead_day.max()==7
    assert not prepared_daily.issued_at.eq(prepared_weekly.issued_at).all()
    monday=pd.Timestamp('2026-11-02')
    selected=sunday_weekly_rows(prepared_weekly,model='weekly_sun__lgbm_stable_recent',
                                location='AL003',monday=monday)
    assert len(selected)==76
    assert sunday_weekly_rows(prepared_weekly,model='weekly_sun__lgbm_stable_recent',
                       location='AL003',monday=pd.Timestamp('2026-11-09')).empty


def test_upcoming_has_no_actual_counts_or_metrics():
    weekly,_=convert_weekly_predictions(sample_week())
    no_actual=weekly.drop(columns=['carwashes'])
    prepared=prepare_upcoming_weekly(no_actual)
    assert len(prepared)==76
    assert 'carwashes' not in prepared
    with pytest.raises(ValueError,match='actual'):
        prepare_upcoming_weekly(weekly)


def test_sunday_issue_strict_calendar():
    assert sunday_issue('2026-11-01')==pd.Timestamp('2026-11-02T00:00Z')
    assert sunday_issue('2026-10-25')==pd.Timestamp('2026-10-25T23:00Z')
    with pytest.raises(ValueError,match='Sunday'):
        sunday_issue('2026-11-02')


def test_training_cutoff_purges_last_sunday_and_future():
    issue=sunday_issue('2026-11-01')
    frame=[]
    for k in range(12):
        old_issue=issue-pd.Timedelta(days=7*(k+1))
        for i in range(30):
            frame.append({'issued_at':old_issue,
                          'timestamp':old_issue+pd.Timedelta(hours=14+i),
                          'lead_day':1+(i//11)})
    df=pd.DataFrame(frame)
    clean=training_cutoff(df,issue,365)
    assert len(clean)==360
    assert (clean.timestamp<issue).all()
    assert (clean.issued_at<issue).all()
    shifted=df.copy()
    shifted.loc[:,'issued_at']+=pd.Timedelta(days=1)
    with pytest.raises(ValueError,match='non-Sunday'):
        training_cutoff(shifted,issue,365)


def test_discover_weekly_output(tmp_path):
    sub=tmp_path/'sunday_weekly_oct_nov'
    sub.mkdir()
    path=sub/'weekly_predictions.csv'
    path.write_text('test')
    assert path in discover_prediction_csvs(tmp_path)


def test_asof_issue_time_stays_at_six_pm_at_dst_change():
    import asof
    for dt in [date(2025,11,2),date(2026,3,8),date(2026,11,1)]:
        issue=asof.issue_time(dt,'America/Chicago',18)
        assert issue.hour==18
        assert issue.date()==dt
    assert asof.issue_time(date(2026,11,1),'America/Chicago',18).tz_convert('UTC') == pd.Timestamp('2026-11-02T00:00Z')
    assert asof.issue_time(date(2026,3,8),'America/Chicago',18).tz_convert('UTC') == pd.Timestamp('2026-03-08T23:00Z')
