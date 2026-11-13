"""Synthetic checks. Never reads or writes user/company datasets."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import demand_feature_ablation as ablation
import demand_model_benchmark as prior
import forecast_reliability as legacy
import recent_demand_features as recent
import reliability_extended as advanced
import storm_features as storm


def small_hourly():
    rows=[]
    for site, actual, forecast in [('AL002',2.,7.),('AL003',80.,91.)]:
        for h in (15,16):
            for model,pred in [('weather_median',actual+1),('lgbm_full_recent',forecast),
                               ('lgbm_no_recent',actual+3)]:
                rows.append({'issued_at':'2026-06-01T00:00:00Z','location_id':site,
                             'timestamp':f'2026-06-02T{h:02d}:00:00Z','model':model,
                             'prediction':pred,'carwashes':actual,
                             'weather_regime':'Rain forecast for target hour',
                             'local_date':'2026-06-02','lead_day':1,
                             'rd_store_age_days':3 if site=='AL002' else 200,
                             'rd_daily_mean_28d':20 if site=='AL002' else 880,
                             'inferred_zero':False})
    return pd.DataFrame(rows)


def test_relative_mae_responds_to_store_scale():
    reports=advanced.make_reports(small_hourly(),comparison='lgbm_no_recent',min_weather_hours=1)
    scores=reports['store_relative_reliability.csv']
    low=scores[(scores.model=='lgbm_full_recent')&(scores.location_id=='AL002')].iloc[0]
    high=scores[(scores.model=='lgbm_full_recent')&(scores.location_id=='AL003')].iloc[0]
    assert low.mae_washes_per_hour==5
    assert low.wape_percent==250
    assert high.mae_washes_per_hour==11
    assert high.wape_percent==pytest.approx(13.75)
    assert low.p90_error_percent_of_store_average==250
    assert high.signed_bias_percent_of_store_average==pytest.approx(13.75)
    assert 'macro_relative_reliability.csv' in reports
    # True first-week observed age, not any age inferred from store ID.
    ages=reports['by_observed_store_age.csv']
    assert {'0-7 days','91-365 days'}.issubset(set(ages.store_age_group))


def test_consistency_metrics_count_issuances_not_hours():
    df=small_hourly()
    second=df.copy()
    second['issued_at']='2026-06-03T00:00:00Z'
    second['timestamp']=pd.to_datetime(second['timestamp'])+pd.Timedelta(days=2)
    second['timestamp']=second['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S%z')
    second['local_date']='2026-06-04'
    df=pd.concat([df,second],ignore_index=True)
    # One issuance has 4 rainy hours and another also has 4.
    report=advanced.make_reports(df,min_weather_hours=4)
    rows=report['weather_consistency_vs_median.csv']
    r=rows[(rows.model=='lgbm_full_recent') & (rows.weather_regime=='Rain forecast for target hour')].iloc[0]
    assert r.issuances_with_at_least_min_hours==2
    assert r.total_hours_in_eligible_issuances==8
    report2=advanced.make_reports(df,min_weather_hours=5)
    r2=report2['weather_consistency_vs_median.csv']
    assert r2.issuances_with_at_least_min_hours.eq(0).all()


def test_full_model_uses_identical_features_to_original_benchmark():
    np.testing.assert_array_equal(ablation.SETS['lgbm_full_recent'],recent.FEATURES)
    assert ablation.input_columns('lgbm_full_recent')==prior._columns(True)
    assert ablation.input_columns('lgbm_no_recent')==prior._columns(False)
    assert not set(ablation.SHORT_LAGS).intersection(ablation.SETS['lgbm_stable_recent'])
    assert 'location_id' in ablation.input_columns('lgbm_stable_recent')[0]
    assert 'rd_store_age_days' in ablation.SETS['lgbm_stable_recent']
    assert ablation.choose_models('weather_median,lgbm_full_recent')==('weather_median','lgbm_full_recent')
    with pytest.raises(ValueError):
        ablation.choose_models('lgbm_full_recent')


def synthetic_examples():
    rows=[]
    first=pd.Timestamp('2026-04-01',tz='America/Chicago')
    for n in range(26):
        day=first+pd.Timedelta(days=n)
        issue=(day-pd.Timedelta(days=1)+pd.Timedelta(hours=18)).tz_convert('UTC')
        for store in ('AL002','AL003'):
            for lead in (0,1):
                date=day+pd.Timedelta(days=lead)
                dow=date.dayofweek
                for hour in range(9 if dow==6 else 8,19):
                    target=(date+pd.Timedelta(hours=hour)).tz_convert('UTC')
                    y=9+(store=='AL003')*12+(hour>12)*8+(date.dayofyear%7)
                    rain=1. if n%7==0 else 0.
                    rows.append({
                        'issued_at':issue,'timestamp':target,'location_id':store,
                        'local_date':str(date.date()),'hour':hour,'dow':dow,
                        'month':date.month,'latitude':33.5,'longitude':-86.,
                        'is_weekend':int(dow>=5),'is_holiday':0,
                        'hour_sin':np.sin(2*np.pi*hour/24),
                        'hour_cos':np.cos(2*np.pi*hour/24),
                        'dow_sin':np.sin(2*np.pi*dow/7),
                        'dow_cos':np.cos(2*np.pi*dow/7),
                        'temperature_2m':22.,'precipitation':rain,
                        'cloud_cover':80. if rain else 20.,
                        'rain_prev24h':0.,'rain_prev72h':4.,'lead_day':lead+1,
                        'carwashes':y,'inferred_zero':False,
                    })
    return pd.DataFrame(rows).sort_values(['issued_at','location_id','timestamp']).reset_index(drop=True)


def test_run_end_to_end_and_resume(tmp_path):
    pytest.importorskip('lightgbm')
    pytest.importorskip('xgboost')
    data=synthetic_examples()
    p=tmp_path/'asof_examples.csv';data.to_csv(p,index=False)
    stm=data[storm.KEYS].copy()
    for col in storm.FEATURES:
        stm[col]=0.
    sp=tmp_path/'storm.csv';stm.to_csv(sp,index=False)
    from backtest import load_examples
    recent_data,_=recent.derive_features(load_examples(p),progress_every=1000)
    rp=tmp_path/'recent.csv';recent_data.to_csv(rp,index=False)
    out=tmp_path/'ablation'
    args=ablation.make_parser().parse_args([
        '--examples',str(p),'--storm-features',str(sp),
        '--recent-demand-features',str(rp),
        '--window-days','90','--validation-from','2026-04-20',
        '--validation-to','2026-04-25','--validation-origins','2',
        '--models','weather_median,xgb_original,lgbm_no_recent,lgbm_full_recent,lgbm_stable_recent',
        '--threads','1','--output',str(out),
    ])
    preds,base,advanced_reports=ablation.run(args)
    assert preds.model.nunique()==5
    assert all(np.isfinite(preds.prediction))
    assert 'macro_relative_reliability.csv' in advanced_reports
    assert 'weather_consistency_vs_no_recent.csv' in advanced_reports
    assert (out/'reliability'/'store_relative_reliability.csv').exists()
    first=(out/'all_predictions.csv').read_bytes()
    ablation.run(args)
    assert (out/'all_predictions.csv').read_bytes()==first
    # Avoid silent mixing of different validation windows with prior results.
    args.validation_origins=1
    with pytest.raises(ValueError,match='different experiment'):
        ablation.run(args)


def test_full_ablation_matches_prior_lightgbm_predictions(tmp_path):
    """The full LightGBM treatment must preserve the previous benchmark."""
    pytest.importorskip('lightgbm')
    data=synthetic_examples()
    p=tmp_path/'examples.csv';data.to_csv(p,index=False)
    extra=data[storm.KEYS].copy()
    for name in storm.FEATURES:
        extra[name]=0.
    sp=tmp_path/'storm.csv';extra.to_csv(sp,index=False)
    from backtest import load_examples
    features,_=recent.derive_features(load_examples(p),progress_every=1000)
    rp=tmp_path/'recent.csv';features.to_csv(rp,index=False)
    common=['--examples',str(p),'--storm-features',str(sp),
            '--recent-demand-features',str(rp),'--window-days','90',
            '--validation-from','2026-04-20','--validation-to','2026-04-25',
            '--validation-origins','2','--threads','1']
    prior_args=prior.make_parser().parse_args(common + [
        '--models','weather_median,lightgbm_recent',
        '--output',str(tmp_path/'prior')])
    previous, _=prior.run(prior_args)
    new_args=ablation.make_parser().parse_args(common + [
        '--models','weather_median,lgbm_full_recent',
        '--output',str(tmp_path/'current')])
    current, _, _=ablation.run(new_args)
    orig=previous[previous.model.eq('lightgbm_recent')].sort_values(storm.KEYS)
    fresh=current[current.model.eq('lgbm_full_recent')].sort_values(storm.KEYS)
    np.testing.assert_allclose(orig.prediction.to_numpy(),fresh.prediction.to_numpy(),atol=1e-5,rtol=0)
