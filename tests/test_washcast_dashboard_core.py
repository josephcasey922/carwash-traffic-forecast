from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
import pytest
from washcast_dashboard_core import (prepare_predictions, combine_predictions,
    metrics, quality_cohort, daily_totals, daily_metrics, paired_predictions,
    week_start, month_summary, discover_prediction_csvs, bucket)


def raw_one_day(day='2026-08-03',model='lgbm_stable_recent',lead=1,zero_hours=()):
    date=pd.Timestamp(day)
    hours=list(range(8,19)) if date.weekday()<6 else list(range(9,19))
    return pd.DataFrame([{
       'issued_at':str(pd.Timestamp(date)-pd.Timedelta(hours=6+24*lead)),
       'timestamp':str(pd.Timestamp(date)+pd.Timedelta(hours=h)),
       'local_date':str(date.date()),'location_id':'AL003','hour':h,
       'lead_day':lead,'model':model,'carwashes':0 if h in zero_hours else 30+h-8,
       'prediction':32+h-8,'inferred_zero':bool(h in zero_hours),
    } for h in hours])


def test_parse_metrics_and_buckets():
    d=prepare_predictions(pd.DataFrame([{
       'issued_at':'2026-08-01T23:00:00Z','timestamp':'2026-08-02T13:00:00Z',
       'local_date':'2026-08-02','location_id':'AL003','hour':8,'lead_day':1,
       'model':'lgbm','carwashes':17,'prediction':18.1,'inferred_zero':'false',
    }]))
    m=metrics(d)
    assert m['n']==1
    assert m['mae']==pytest.approx(1.1)
    assert m['bucket_agreement']==0
    assert m['bucket_over']==100


def test_deduplicates_identical_overlap_but_catches_conflicts():
    a=prepare_predictions(raw_one_day())
    d,n=combine_predictions([a,a.copy()])
    assert n==len(a) and len(d)==len(a)
    b=a.copy();b.loc[b.index[0],'prediction']+=1
    with pytest.raises(ValueError,match='different forecasts'):
        combine_predictions([a,b])
    c=a.copy();c.loc[c.index[0],'carwashes']+=1
    with pytest.raises(ValueError,match='Conflicting actual'):
        combine_predictions([a,c])


def test_daily_totals_only_complete_operating_days():
    mon=prepare_predictions(raw_one_day('2026-08-03'))
    sun=prepare_predictions(raw_one_day('2026-08-02'))
    d=daily_totals(pd.concat([mon,sun],ignore_index=True))
    assert d.complete.sum()==2
    assert set(d.expected_hours)=={10,11}
    assert daily_metrics(d)['days']==2
    partial=daily_totals(mon.iloc[:-2]);assert not partial.iloc[0].complete
    assert daily_metrics(partial)['days']==0


def test_quality_filters_not_corrupt_targets():
    a=prepare_predictions(raw_one_day(zero_hours=[9,10,11,12]))
    assert len(quality_cohort(a,'All (official)'))==11
    assert len(quality_cohort(a,'Recorded hours only'))==7
    assert len(quality_cohort(a,'Store-days with no inferred zeros'))==0
    assert len(quality_cohort(a,'Exclude store-days with 4+ inferred zeros'))==0
    assert int(a.inferred_zero.sum())==4


def test_paired_models_require_identical_targets():
    a=prepare_predictions(raw_one_day())
    b=prepare_predictions(raw_one_day(model='weather_median'))
    p=paired_predictions(pd.concat([a,b]),'lgbm_stable_recent','weather_median')
    assert len(p)==11
    b.loc[b.index[0],'carwashes']=999
    with pytest.raises(ValueError,match='same actual'):
        paired_predictions(pd.concat([a,b]),'lgbm_stable_recent','weather_median')


def test_week_start_month_summary_and_discovery(tmp_path):
    a=prepare_predictions(raw_one_day())
    assert week_start(a.local_date).iloc[0]==pd.Timestamp('2026-08-03')
    assert month_summary(a).iloc[0]['month']=='2026-08'
    p=tmp_path/'demand_ablation_october'/'all_predictions.csv'
    p.parent.mkdir();p.touch()
    assert discover_prediction_csvs(tmp_path)==[p]


def test_rejects_bad_source():
    with pytest.raises(ValueError,match='missing columns'):
        prepare_predictions(pd.DataFrame({'model':['a']}))


def test_adaptation_format_uses_adjusted_forecasts_and_chicago_dst():
    rows=[]
    for issued,target in [('2026-05-01 23:00:00+00:00','2026-05-02'),
                          ('2026-11-01 00:00:00+00:00','2026-11-01')]:
        # Nov issuance = Oct 31, 19:00 Chicago before DST ends Nov 1.
        for candidate,adj in [('base',20.0),('store_fast_14d',23.0)]:
            rows.append(dict(issued_at=issued,local_date=target,location_id='AL003',
                             hour=9,carwashes=22,prediction=10.0,
                             adjusted_prediction=adj,candidate=candidate,inferred_zero=False))
    import pandas as pd
    d=prepare_predictions(pd.DataFrame(rows),source='adaptation_example')
    assert set(d.model)=={'adaptation_base','adaptation_store_fast_14d'}
    assert set(d.lead_day)=={1}
    assert d.loc[d.model.eq('adaptation_store_fast_14d'),'prediction'].eq(23).all()
    # UTC timestamp respects DST (9 AM CDT=14:00Z; 9 AM CST=15:00Z).
    assert set(d.timestamp.dt.hour)=={14,15}
    assert d.loc[d.model.eq('adaptation_store_fast_14d'),'abs_error'].eq(1).all()


def test_adaptation_rejects_non_next_day_and_unrelated_reports():
    d=pd.DataFrame([dict(issued_at='2026-05-01 23:00:00+00:00',
                        local_date='2026-05-04',location_id='AL003',hour=10,
                        carwashes=22,adjusted_prediction=23,candidate='store_fast_14d')])
    with pytest.raises(ValueError,match='not next-day'):
        prepare_predictions(d)
    with pytest.raises(ValueError,match='not a prediction file'):
        prepare_predictions(pd.DataFrame([{'candidate':'base','mae':4.5}]))


def test_real_adaptation_csv_shape_if_available():
    path=Path('/mnt/data/washcast_adaptation_aug61_exploratory/all_candidate_day1_predictions.csv')
    if path.is_file():
        sampled=pd.read_csv(path,nrows=100)
        out=prepare_predictions(sampled,'adaptation_aug61')
        assert out.lead_day.eq(1).all()
        assert out.timestamp.notna().all()
        assert (out.prediction-out.adjusted_prediction).abs().max()<1e-10


def test_labor_alignment_exact_company_formula_not_just_bucket_agreement():
    # Actual recommendations: [2,3,4,5]; forecast recommendations: [2,3,3,4].
    # 2 employee-hour errors out of 14 recommended -> 85.714...%.
    raw=pd.DataFrame([
        dict(issued_at='2026-10-01T23:00:00Z',timestamp=f'2026-10-02T{13+i:02d}:00:00Z',
             local_date='2026-10-02',location_id='AL003',hour=8+i,lead_day=1,
             model='lgbm_stable_recent',carwashes=y,prediction=p,inferred_zero=False)
        for i,(y,p) in enumerate([(10,10),(20,20),(50,35),(65,50)])
    ])
    d=prepare_predictions(raw)
    m=metrics(d)
    assert m['bucket_agreement']==50
    assert m['forecast_labor_alignment_percent']==pytest.approx(100*(1-2/14))
    assert m['recommended_employee_hours']==14
    assert m['absolute_employee_hour_difference']==2
    assert m['under_recommended_employee_hours']==2
    assert m['over_recommended_employee_hours']==0
    assert d.employee_difference.tolist()==[0,0,-1,-1]


def test_rounding_half_up_at_boundaries_matches_alignment_report():
    from washcast_dashboard_core import employee_counts
    assert employee_counts([17.49,17.50,40.49,40.50,60.49,60.50]).tolist()==[2,3,3,4,4,5]
    # Check both the derived recommendation and the legacy chart bucket.
    assert bucket(pd.Series([17.5,40.5,60.5])).tolist()==[1,2,3]


def test_labor_alignment_aggregates_employee_hours_and_respects_subset():
    a=prepare_predictions(raw_one_day())
    b=prepare_predictions(raw_one_day('2026-08-04'))
    all_df=pd.concat([a,b],ignore_index=True)
    m=metrics(all_df)
    total_difference=(all_df.predicted_recommended_employees-all_df.actual_recommended_employees).abs().sum()
    actual_employee_hours=all_df.actual_recommended_employees.sum()
    assert m['forecast_labor_alignment_percent']==pytest.approx(100*(1-total_difference/actual_employee_hours))
    m_one=metrics(a)
    assert m_one['recommended_employee_hours']==a.actual_recommended_employees.sum()
    assert metrics(a.iloc[:0])['forecast_labor_alignment_percent']!=metrics(a.iloc[:0])['forecast_labor_alignment_percent']
    # Metrics recompute on filters; no static imported scores.
    contaminated=prepare_predictions(raw_one_day(zero_hours=(8,9)))
    clean=quality_cohort(contaminated,'Recorded hours only')
    assert metrics(clean)['n']==9
    assert metrics(contaminated)['n']==11


def test_report_reconciliation_against_uploaded_store_month_summaries():
    # These are aggregate CSVs only; recompute the official rollup using the
    # sum of differences / sum of recommended employee-hours, not mean of scores.
    path=Path('/mnt/data/by_store_month.csv')
    if not path.exists():
        return
    report=pd.read_csv(path)
    for (_, _),g in report.groupby(['model','month']):
        correct=100*(1-g.absolute_employee_hour_difference.sum()/g.recommended_employee_hours.sum())
        assert 0<=correct<=100
        assert g.hours.sum()>0
    stable_oct=report[(report.model=='lgbm_stable_recent')&(report.month=='2026-10')]
    assert 100*(1-stable_oct.absolute_employee_hour_difference.sum()/stable_oct.recommended_employee_hours.sum())==pytest.approx(91.714,abs=.001)
