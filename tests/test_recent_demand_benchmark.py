"""Self-contained synthetic tests. Never reads or modifies company source data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import backtest
import storm_features as storm
import recent_demand_features as recent
import demand_model_benchmark as bench
import forecast_reliability as reliability


def synthetic_examples(issuances=34, sites=('AL002','AL003'), gap_day=None):
    rows=[]
    # All forecast decisions are 6 PM Chicago, predict the following two days.
    first=pd.Timestamp('2026-04-01',tz='America/Chicago')
    for n in range(issuances):
        day=first+pd.Timedelta(days=n)
        issue=(day-pd.Timedelta(days=1)+pd.Timedelta(hours=18)).tz_convert('UTC')
        for site in sites:
            for lead in (0,1):
                target_day=(day+pd.Timedelta(days=lead))
                if gap_day is not None and target_day.date()==gap_day:
                    continue
                dow=target_day.dayofweek
                for hour in range(9 if dow==6 else 8,19):
                    target=(target_day+pd.Timedelta(hours=hour)).tz_convert('UTC')
                    count=max(0,int(15 + (site=='AL003')*8 + hour-10 + ((target_day.dayofyear)%5)*3))
                    rain=float((n%8)==0)
                    rows.append({'issued_at':issue,'timestamp':target,
                        'location_id':site,'local_date':str(target_day.date()),
                        'hour':hour,'dow':dow,'month':target_day.month,
                        'latitude':33.5,'longitude':-86.0,
                        'is_weekend':int(dow>=5),'is_holiday':0,
                        'hour_sin':np.sin(2*np.pi*hour/24),'hour_cos':np.cos(2*np.pi*hour/24),
                        'dow_sin':np.sin(2*np.pi*dow/7),'dow_cos':np.cos(2*np.pi*dow/7),
                        'temperature_2m':24.,'precipitation':rain,
                        'cloud_cover':80. if rain else 20.,'rain_prev24h':0.,'rain_prev72h':3.,
                        'lead_day':lead+1,'carwashes':count,'inferred_zero':False})
    return pd.DataFrame(rows).sort_values(['issued_at','location_id','timestamp']).reset_index(drop=True)


def test_dedup_inconsistent_labels():
    ex=synthetic_examples(9)
    hist=recent.unique_history(ex)
    assert len(hist)<len(ex)
    key=ex.loc[0,['location_id','timestamp']]
    matching=ex.index[ex.location_id.eq(key.location_id)&ex.timestamp.eq(key.timestamp)]
    if len(matching)>1:
        ex.loc[matching[-1],'carwashes']+=100
        with pytest.raises(ValueError,match='Inconsistent'):
            recent.unique_history(ex)


def test_asof_features_do_not_see_future_labels_or_inferred_zeros():
    ex=synthetic_examples(25)
    one,_=recent.derive_features(ex,progress_every=1000)
    focus=ex.issued_at.unique()[12]
    # Change target outcomes after focus issue; history at focus must be unchanged.
    later=ex.copy()
    later.loc[later.timestamp >= focus,'carwashes']+=1000
    two,_=recent.derive_features(later,progress_every=1000)
    cut=ex.issued_at.eq(focus)
    pd.testing.assert_frame_equal(one.loc[cut].reset_index(drop=True),
                                  two.loc[cut].reset_index(drop=True))
    # A 7-day historical lag for a 2-day-ahead target is always strictly prior to issuance.
    at=one.loc[cut,'rd_lag_7d'].to_numpy()
    assert np.isfinite(at).all()


def test_absent_store_day_is_not_zero():
    missing=pd.Timestamp('2026-04-12').date()
    ex=synthetic_examples(26,gap_day=missing)
    hist=recent.unique_history(ex)
    days=recent.summarize_days(hist)
    assert not (days.day==np.datetime64(missing)).any()
    result,_=recent.derive_features(ex,progress_every=1000)
    assert result['rd_daily_days_7d'].max() <= 7


def test_unknown_cold_start_location():
    ex=synthetic_examples(23,sites=('AL002','AL003'))
    # A new store with one fresh issued forecast but no completed history.
    issue=ex.issued_at.max()
    rows=ex[(ex.issued_at==issue)&(ex.location_id=='AL002')].copy()
    rows['location_id']='AL016'
    ex=pd.concat([ex,rows],ignore_index=True).sort_values(
        ['issued_at','location_id','timestamp']).reset_index(drop=True)
    result,_=recent.derive_features(ex,progress_every=1000)
    newborn=result[result.location_id.eq('AL016')]
    assert newborn.rd_daily_days_28d.eq(0).all()
    assert newborn.rd_store_age_days.eq(0).all()
    assert newborn.rd_dow_hour_median_56d.isna().all()
    assert newborn.rd_network_dow_hour_median_56d.notna().all()


def test_reliability_skill_and_paired_checks():
    d=pd.DataFrame([
      {'issued_at':'2026-06-01T00:00:00Z','location_id':'AL002',
       'timestamp':f'2026-06-02T{h:02d}:00:00Z','model':m,
       'prediction':pred,'carwashes':10,'weather_regime':'Otherwise dry',
       'local_date':'2026-06-01','lead_day':1,'rd_store_age_days':100}
      for h in (15,16) for m,pred in [('weather_median',14),('xgb_recent',12)]
    ])
    reports=reliability.make_reports(d)
    score=reports['skill_vs_weather_median.csv'].iloc[0]
    assert score['skill_percent']==pytest.approx(50.)
    changed=d.copy()
    changed.loc[(changed.model=='xgb_recent'),'carwashes']=11
    with pytest.raises(ValueError,match='identical'):
        reliability.make_reports(changed)


def test_benchmark_synthetic_end_to_end(tmp_path):
    pytest.importorskip('xgboost')
    pytest.importorskip('lightgbm')
    ex=synthetic_examples(25)
    src=tmp_path/'examples.csv'
    ex.to_csv(src,index=False)
    storm_df=ex[recent.KEYS].copy()
    for c in storm.FEATURES:
        storm_df[c]=0.
    storm_file=tmp_path/'storm.csv'
    storm_df.to_csv(storm_file,index=False)
    from backtest import load_examples
    rec,info=recent.derive_features(load_examples(src),progress_every=1000)
    assert info['unique_store_hours']>100
    recfile=tmp_path/'recent.csv';rec.to_csv(recfile,index=False)
    p=bench.make_parser()
    args=p.parse_args(['--examples',str(src),'--storm-features',str(storm_file),
         '--recent-demand-features',str(recfile),'--window-days','90',
         '--validation-from','2026-04-18','--validation-to','2026-04-24',
         '--validation-origins','2','--threads','1',
         '--models','weather_median,adaptive_median,xgb_original,xgb_recent,lightgbm_recent,poisson_recent',
         '--output',str(tmp_path/'results')])
    pred,reports=bench.run(args)
    assert pred.model.nunique()==6
    assert len(reports['by_location.csv'])==12
    assert (pred.prediction>=0).all()
    # A repeated invocation must reuse completed fold files.
    before=(tmp_path/'results'/'all_predictions.csv').read_bytes()
    bench.run(args)
    assert before==(tmp_path/'results'/'all_predictions.csv').read_bytes()


def test_original_xgboost_design_matrix_matches_existing_storm_experiment():
    """Control variant may not quietly change preprocessing or feature set."""
    import storm_loss_experiment as old
    import experiments
    ex=experiments.add_season_features(synthetic_examples(5))
    for f in storm.FEATURES:
        ex[f]=0.
    original=old.inputs(ex,'storm')
    new=bench._inputs(ex,False)
    pd.testing.assert_frame_equal(original,new)
    old_transform=old.transformer('storm')
    new_transform=bench._transformer(False)
    np.testing.assert_allclose(old_transform.fit_transform(original),
                               new_transform.fit_transform(new),atol=1e-10)
