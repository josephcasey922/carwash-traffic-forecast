"""v5: verify hypothetical employee-hour formula and FAIR model-labelled comparisons."""
import numpy as np
import pandas as pd
import pytest

from washcast_dashboard_core import (
    metrics,prepare_predictions,matched_model_alignment,matched_alignment_by_period,
)


def sample_rows(actuals, predictions, dates, *, model, origin='daily'):
    rows=[]
    for k,(a,p,day) in enumerate(zip(actuals,predictions,dates)):
        local=pd.Timestamp(day)
        hour=8+k%7
        target=(local+pd.Timedelta(hours=hour)).tz_localize('America/Chicago').tz_convert('UTC')
        if origin=='weekly':
            sunday=local-pd.Timedelta(days=(local.dayofweek+1)%7)
            issue=(sunday+pd.Timedelta(hours=18)).tz_localize('America/Chicago').tz_convert('UTC')
            lead=(local-sunday).days
        else:
            issue=(local-pd.Timedelta(days=1)+pd.Timedelta(hours=18)).tz_localize('America/Chicago').tz_convert('UTC')
            lead=1
        rows.append(dict(issued_at=issue,location_id='AL003',timestamp=target,
                         local_date=local.strftime('%Y-%m-%d'),hour=hour,
                         lead_day=lead,carwashes=a,model=model,prediction=p,
                         inferred_zero=False))
    return prepare_predictions(pd.DataFrame(rows))


def test_daily_uses_hourly_abs_differences_not_aggregate_or_mean():
    d=sample_rows([10,20,50,65],[10,20,35,50],['2026-11-02']*4,model='daily_a')
    # Actual staffing [2,3,4,5] vs proposed [2,3,3,4]
    m=metrics(d)
    assert m['recommended_employee_hours']==14
    assert m['absolute_employee_hour_difference']==2
    assert m['forecast_labor_alignment_percent']==pytest.approx(100*(1-2/14))
    assert m['bucket_agreement']==50


def test_week_is_weighted_total_employee_hours_not_average_of_daily_scores():
    mon=pd.Timestamp('2026-11-02')
    dates=[(mon+pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
    # Monday: 2 actual rec employees, model 3 => one discrepancy / 2 (50% score)
    # Tue-Sun: 5 actual rec employees and 5 predicted => 100% each
    d=sample_rows([10]+[65]*6,[18]+[65]*6,dates,model='daily_a')
    week=metrics(d)
    assert week['recommended_employee_hours']==32
    assert week['absolute_employee_hour_difference']==1
    assert week['forecast_labor_alignment_percent']==pytest.approx(96.875)
    naive=(50+100*6)/7
    assert week['forecast_labor_alignment_percent']!=pytest.approx(naive)
    by_day=matched_alignment_by_period(d,['daily_a'],period='day')
    by_week=matched_alignment_by_period(d,['daily_a'],period='week')
    assert by_day.shape[0]==7
    assert by_week.shape[0]==1
    assert by_week.iloc[0].forecast_labor_alignment_percent==pytest.approx(week['forecast_labor_alignment_percent'])
    assert by_week.iloc[0]['period']=='2026-11-02'


def test_matched_weekly_vs_daily_uses_target_hours_not_issue_time():
    # Sunday 18:00 issued weekly forecast vs each previous-evening daily forecast
    dates=['2026-11-02','2026-11-03','2026-11-04']
    daily=sample_rows([17,42,64],[19,45,59],dates,model='daily_stable')
    weekly=sample_rows([17,42,64],[17,40,63],dates,
                       model='weekly_sun__daily_stable',origin='weekly')
    # Drop a weekly target hour; fair comparison must also drop that hour from daily.
    weekly=weekly.iloc[:2]
    combined=pd.concat([daily,weekly],ignore_index=True)
    table=matched_model_alignment(combined,['daily_stable','weekly_sun__daily_stable'],
                                  match_target_hours=True)
    assert table.shape[0]==2
    assert table.hours.tolist()==[2,2]
    assert table.recommended_employee_hours.tolist()==[6,6]
    assert table.loc[0,'absolute_employee_hour_difference']==1
    assert table.loc[1,'absolute_employee_hour_difference']==1
    # Monday Day 1 happens to share the Sunday 6 PM issue. Tuesday onwards do not.
    # Issue-based pairing therefore covers ONLY Monday, while target-hour pairing covers both.
    issue_paired=matched_model_alignment(combined,['daily_stable','weekly_sun__daily_stable'],
                                   match_target_hours=False)
    assert issue_paired.hours.tolist()==[1,1]


def test_three_model_comparison_same_denominator_and_correct_labels():
    dates=['2026-11-02','2026-11-03']
    a=sample_rows([17,65],[17,65],dates,model='daily_a')
    b=sample_rows([17,65],[18,41],dates,model='daily_b')
    c=sample_rows([17,65],[15,61],dates,model='weekly_sun__daily_a',origin='weekly')
    combined=pd.concat([a,b,c],ignore_index=True)
    result=matched_model_alignment(combined,['weekly_sun__daily_a','daily_a','daily_b'],
                                    match_target_hours=True)
    assert result.model.tolist()==['weekly_sun__daily_a','daily_a','daily_b']
    assert result.recommended_employee_hours.nunique()==1
    assert result.hours.tolist()==[2,2,2]
    assert result.iloc[1].forecast_labor_alignment_percent==100
    assert result.iloc[0].forecast_labor_alignment_percent==100
    assert result.iloc[2].forecast_labor_alignment_percent<100
    by_date=matched_alignment_by_period(combined,result.model.tolist(),
                                         period='day',match_target_hours=True)
    assert len(by_date)==6


def test_mismatched_labels_fail_loudly():
    a=sample_rows([17], [18], ['2026-11-02'],model='daily_a')
    b=sample_rows([19], [18], ['2026-11-02'],model='daily_b')
    with pytest.raises(ValueError,match='disagree on actual washes'):
        matched_model_alignment(pd.concat([a,b]),['daily_a','daily_b'])


def test_duplicate_forecast_hours_fail_loudly():
    a=sample_rows([17],[18],['2026-11-02'],model='daily_a')
    with pytest.raises(ValueError,match='duplicate forecast keys'):
        matched_model_alignment(pd.concat([a,a]),['daily_a'])
