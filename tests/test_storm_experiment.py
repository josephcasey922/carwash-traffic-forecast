"""Synthetic-only tests: no company data or network access required."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backtest
import experiments
import model_benchmark
import storm_features as sf
import storm_loss_experiment as sl


def demo_data(n_issues=14):
    issues = [pd.Timestamp('2026-05-01T23:00Z') + pd.Timedelta(days=i * 7)
              for i in range(n_issues)]
    sites = [('A', 33.6), ('B', 33.8)]
    history_rows = []
    hourly = pd.date_range(issues[0] - pd.Timedelta(days=6),
                           issues[-1] + pd.Timedelta(days=5), freq='h')
    for site, lat in sites:
        for hour in hourly:
            history_rows.append({'location_id': site, 'timestamp': hour,
                                 'temperature_2m': 15 + hour.hour / 4,
                                 'precipitation': 1.0 if (hour.dayofyear % 6 == 0 and hour.hour in (6,7,8)) else 0.,
                                 'cloud_cover': 20 + hour.hour})
    history = pd.DataFrame(history_rows)
    archives, examples = [], []
    for i, issue in enumerate(issues):
        for site, lat in sites:
            for offset in range(1, 50):
                target = issue + pd.Timedelta(hours=offset)
                future_rain = 1.5 if (offset + i) % 13 in (0,1,2,3) else 0.
                archives.append({'location_id': site, 'issued_at': issue,
                                 'timestamp': target, 'temperature_2m': 22 + .1 * offset,
                                 'precipitation': future_rain, 'cloud_cover': 25 + (offset % 10) * 5})
                if offset > 24:
                    continue
                local = target.tz_convert('America/Chicago')
                examples.append({'issued_at': issue, 'timestamp': target,
                                 'local_date': str(local.date()), 'location_id': site,
                                 'latitude': lat, 'longitude': -86.5,
                                 'hour': local.hour, 'dow': local.dayofweek, 'month': local.month,
                                 'is_weekend': int(local.dayofweek >= 5), 'is_holiday': 0,
                                 'hour_sin': np.sin(local.hour * np.pi / 12),
                                 'hour_cos': np.cos(local.hour * np.pi / 12),
                                 'dow_sin': np.sin(local.dayofweek * 2 * np.pi / 7),
                                 'dow_cos': np.cos(local.dayofweek * 2 * np.pi / 7),
                                 'temperature_2m': 22 + .1 * offset,
                                 'precipitation': future_rain,
                                 'cloud_cover': 25 + (offset % 10) * 5,
                                 'rain_prev24h': float(i % 4 + offset * .03),
                                 'rain_prev72h': float(i % 7 + offset * .08),
                                 'lead_day': 1,
                                 'carwashes': float(max(0, 8 + (i + offset) % 11 - 4 * future_rain)),
                                 'inferred_zero': False})
    return pd.DataFrame(examples), history, pd.DataFrame(archives)


def test_storm_recent_hours_duration_and_no_current_hour_leak():
    hours = pd.date_range('2026-01-01T00:00Z', periods=110, freq='h')
    rain = np.zeros(len(hours))
    rain[96:99] = [1.0, 3.0, 0.8]
    weather = pd.DataFrame({'precipitation': rain, 'temperature_2m': np.arange(len(hours)),
                            'cloud_cover': np.arange(len(hours))}, index=hours)
    out = sf.derive_hourly_features(weather)
    # Hour 99: three rainy hours strictly before current hour.
    assert out.iloc[99].storm_hours_since_rain == 1
    assert out.iloc[99].storm_last_event_duration_h == 3
    assert out.iloc[99].storm_rain_ended_this_hour == 1
    assert out.iloc[99].storm_rain_prev3h_mm == pytest.approx(4.8)
    # Hour 96 rainfall has only just begun. All lagged rain features are dry.
    assert out.iloc[96].storm_rain_prev3h_mm == 0
    assert out.iloc[96].storm_rain_ended_this_hour == 0
    assert out.iloc[95].storm_hours_since_rain == 73
    weather.loc[hours[98], 'precipitation'] = np.nan
    missing = sf.derive_hourly_features(weather)
    assert np.isnan(missing.iloc[99].storm_rain_prev3h_mm)
    assert np.isnan(missing.iloc[99].storm_hours_since_rain)


def test_future_realized_weather_and_later_vintages_cannot_leak():
    sample, history, archive = demo_data(2)
    issue = sample.issued_at.iloc[0]
    target = sample.iloc[[1]].copy()
    base = sf.enrich_examples(target, history, archive, progress_every=0)
    poisoned_history = history.copy()
    poisoned_history.loc[poisoned_history.timestamp > issue, 'precipitation'] = 5000.
    result = sf.enrich_examples(target, poisoned_history, archive, progress_every=0)
    np.testing.assert_allclose(base[list(sf.FEATURES)], result[list(sf.FEATURES)], equal_nan=True)
    archive_other_issue = archive.copy()
    archive_other_issue.loc[archive_other_issue.issued_at != issue, 'precipitation'] = 8000.
    result = sf.enrich_examples(target, history, archive_other_issue, progress_every=0)
    np.testing.assert_allclose(base[list(sf.FEATURES)], result[list(sf.FEATURES)], equal_nan=True)
    future = archive.copy()
    future.loc[(future.issued_at == issue) &
               (future.timestamp > target.timestamp.iloc[0]), 'precipitation'] = 1000.
    result = sf.enrich_examples(target, history, future, progress_every=0)
    np.testing.assert_allclose(base[list(sf.FEATURES)], result[list(sf.FEATURES)], equal_nan=True)


def test_missing_archived_target_and_mismatch_fail():
    example, history, archive = demo_data(1)
    one = example.iloc[[0]]
    key = (archive.issued_at == one.issued_at.iloc[0]) & (archive.location_id == one.location_id.iloc[0]) & (archive.timestamp == one.timestamp.iloc[0])
    with pytest.raises(ValueError, match='target weather'):
        sf.enrich_examples(one, history, archive.loc[~key], progress_every=0)
    bad = archive.copy()
    bad.loc[key, 'precipitation'] += 10
    with pytest.raises(ValueError, match='does not match'):
        sf.enrich_examples(one, history, bad, progress_every=0)


def test_original_boosted_poisson_reproduces_prior_pipeline():
    raw, history, archive = demo_data()
    data = experiments.add_season_features(raw)
    issue = sorted(data.issued_at.unique())[10]
    train, test = backtest.fold_sets(data, issue, 90)
    assert len(train) > 200 and len(test) > 0
    # No storm features involved in original model.
    ref = experiments.boost_model(train, True)
    p0 = ref.predict(experiments.inputs(test, True))
    trans = sl.transformer('original')
    xt = trans.fit_transform(sl.inputs(train, 'original'))
    xv = trans.transform(sl.inputs(test, 'original'))
    estimator = sl.make_estimator('boosted','poisson')
    estimator.fit(xt, train.carwashes.to_numpy(dtype=float))
    p1 = sl.predict(estimator,'boosted','poisson',xv)
    np.testing.assert_allclose(p0,p1,rtol=1e-10,atol=1e-10)


@pytest.mark.parametrize('model', ['boosted','xgboost','catboost','lightgbm'])
@pytest.mark.parametrize('loss', ['poisson','absolute'])
def test_each_loss_has_finite_nonnegative_predictions(model, loss):
    raw, _, _ = demo_data(8)
    raw = experiments.add_season_features(raw)
    trans = sl.transformer('original')
    data = trans.fit_transform(sl.inputs(raw, 'original'))
    labels = raw.carwashes.to_numpy(float)
    estimator = sl.make_estimator(model, loss, threads=2)
    x = pd.DataFrame(data,columns=trans.get_feature_names_out()) if model == 'lightgbm' else data
    estimator.fit(x, labels)
    p = sl.predict(estimator, model, loss, x[:12])
    assert len(p) == 12 and np.isfinite(p).all() and (p >= 0).all()


def test_offline_e2e_and_resume(tmp_path):
    example, history, archive = demo_data()
    ex_path = tmp_path / 'examples.csv'
    sf_path = tmp_path / 'storm.csv'
    example.to_csv(ex_path,index=False)
    extras = sf.enrich_examples(backtest.load_examples(ex_path), history, archive, progress_every=0)
    extras.to_csv(sf_path,index=False)
    args = argparse.Namespace(examples=str(ex_path), storm_features=str(sf_path),
                              selected_configs=None, window_days=90,
                              validation_from='2026-07-01', validation_to='2026-07-31',
                              validation_origins=2, models='boosted,xgboost',
                              losses='poisson,absolute', feature_sets='original,storm',
                              threads=2, output=str(tmp_path/'results'))
    pred, reports = sl.evaluate(args)
    assert pred.variant.nunique() == 10
    assert pred.groupby('variant').size().nunique() == 1
    assert set(['overall_metrics.csv','by_weather_regime.csv','gain_vs_reference.csv']).issubset(reports)
    assert reports['overall_metrics.csv'].shape[0] == 10
    baseline_count = len(list((tmp_path/'results'/'fold_predictions').rglob('*.csv')))
    result2, _ = sl.evaluate(args)
    assert len(result2) == len(pred)
    assert baseline_count == len(list((tmp_path/'results'/'fold_predictions').rglob('*.csv')))
    args.threads = 1
    with pytest.raises(ValueError, match='different run'):
        sl.evaluate(args)
