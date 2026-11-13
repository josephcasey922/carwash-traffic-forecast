"""Lightweight checks of offline, resumable global-model tuning (synthetic only)."""
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
import model_tuning as mt


def demo_frame(n_issues=14):
    items = []
    for i in range(n_issues):
        issued = pd.Timestamp('2026-05-01T23:00:00Z') + pd.Timedelta(days=7 * i)
        for site, lat in [('A', 33.6), ('B', 33.8)]:
            for hour_offset in range(1, 25):
                target = issued + pd.Timedelta(hours=hour_offset)
                local = target.tz_convert('America/Chicago')
                rain = float((i + hour_offset) % 5 == 0)
                items.append({
                    'issued_at': issued, 'timestamp': target,
                    'local_date': str(local.date()), 'location_id': site,
                    'latitude': lat, 'longitude': -86.5,
                    'hour': local.hour, 'dow': local.dayofweek, 'month': local.month,
                    'is_weekend': int(local.dayofweek >= 5), 'is_holiday': 0,
                    'hour_sin': np.sin(local.hour * np.pi / 12),
                    'hour_cos': np.cos(local.hour * np.pi / 12),
                    'dow_sin': np.sin(local.dayofweek * 2 * np.pi / 7),
                    'dow_cos': np.cos(local.dayofweek * 2 * np.pi / 7),
                    'temperature_2m': 18 + 5 * np.sin(i), 'precipitation': rain,
                    'cloud_cover': 30 + 30 * rain, 'rain_prev24h': float(i % 4),
                    'rain_prev72h': float(i % 7), 'lead_day': 1,
                    'carwashes': float(max(0, 8 + (i + hour_offset) % 11 - 4 * rain)),
                    'inferred_zero': False,
                })
    return pd.DataFrame(items)


def test_trial_generation_reproducible_model_independent():
    all_specs = mt.trial_specs(('boosted','catboost','xgboost'), 4, 23)
    cb_only = mt.trial_specs(('catboost',), 4, 23)
    assert [x for x in all_specs if x['model']=='catboost'] == cb_only
    assert all_specs[0]['candidate_id'] == 'default'
    assert all_specs[0]['overrides'] == {}
    assert len(all_specs) == 12
    assert mt.trial_specs(('boosted',), 4, 23) == mt.trial_specs(('boosted',), 4, 23)
    with pytest.raises(ValueError):
        mt.parse_models('boosted,boosted')


def test_selection_evenly_spaced_and_nonleaky(tmp_path):
    example_file = tmp_path / 'examples.csv'
    demo_frame().to_csv(example_file, index=False)
    data = backtest.load_examples(example_file)
    origins = mt.select_validation_issues(data, '2026-06-01', '2026-07-31', 3)
    assert len(origins) == 3
    assert origins == sorted(origins)
    for issue in origins:
        train, test = backtest.fold_sets(data, issue, 90)
        assert (train.timestamp < issue).all()
        assert (train.issued_at < issue).all()
        assert (test.issued_at == issue).all()


def test_weather_adjusted_median_learns_only_training_weather():
    train = pd.DataFrame([
        {'location_id': 'A', 'dow': 2, 'hour': 10, 'carwashes': 20.,
         'precipitation': 0., 'rain_prev72h': 0.},
        {'location_id': 'A', 'dow': 2, 'hour': 10, 'carwashes': 2.,
         'precipitation': 1., 'rain_prev72h': 0.},
        {'location_id': 'A', 'dow': 2, 'hour': 10, 'carwashes': 2.,
         'precipitation': 1., 'rain_prev72h': 0.},
        {'location_id': 'A', 'dow': 2, 'hour': 10, 'carwashes': 20.,
         'precipitation': 0., 'rain_prev72h': 0.},
    ])
    model = mt.WeatherAdjustedMedian(local_shrink=0, global_shrink=0).fit(train)
    test = train.iloc[[0,1]].copy()
    p = model.predict(test)
    assert len(p) == 2
    assert p[0] > p[1]
    assert np.isfinite(p).all() and (p >= 0).all()


def test_sqlite_resume_and_changed_run_guard(tmp_path):
    src = tmp_path/'ex.csv'
    src.write_text('a,b\n1,2\n')
    obj = {'source': mt.source_fingerprint(src), 'models': ['boosted'], 'params': {'x': 1}}
    folder = tmp_path/'run'
    with mt.prepare_output(folder, obj) as db:
        row = {'issued_at':'2026-06-01', 'model':'boosted','candidate_id':'default',
               'scores':{'all':{'n':1,'sum_abs':3.,'sum_sq':9., 'sum_signed':3.,'sum_actual':1.}}}
        mt.save_result(db, row)
        assert mt.finished_keys(db) == {('2026-06-01', 'boosted', 'default')}
    with mt.prepare_output(folder, obj) as db:
        assert mt.finished_keys(db) == {('2026-06-01', 'boosted', 'default')}
    with pytest.raises(ValueError, match='different tuning run'):
        mt.prepare_output(folder, {**obj, 'params': {'x': 2}})


def test_offline_end_to_end_and_resume(tmp_path):
    file = tmp_path/'examples.csv'
    demo_frame().to_csv(file, index=False)
    args = argparse.Namespace(examples=str(file), window_days=90,
                              validation_from='2026-07-01', validation_to='2026-07-31',
                              validation_origins=2, models='boosted,catboost',
                              candidates_per_model=1, seed=42, threads=2,
                              no_weather_median=False, output=str(tmp_path/'run'))
    summary, chosen = mt.run(args)
    assert set(summary.model) == {'boosted','catboost','baseline','weather_median'}
    assert summary.eligible.all()
    assert set(chosen['selected_models']) == {'boosted','catboost'}
    assert chosen['validation_only']
    assert len(pd.read_csv(tmp_path/'run'/'tuning_fold_scores.csv')) == 8
    summary_again, _ = mt.run(args)
    assert len(summary_again) == len(summary)
    with mt.prepare_output(tmp_path/'run', json.loads((tmp_path/'run'/'tuning_manifest.json').read_text())) as db:
        assert len(mt.load_results(db)) == 8
