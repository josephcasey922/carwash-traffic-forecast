"""Synthetic-only tests for pooled vs one-store global-XGBoost comparison."""
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
import baseline_features as bf
import baseline_informed_experiment as bi
import location_model_experiment as lm
import model_tuning as tuning
import storm_features as sf


def fixture_files(tmp_path):
    rows = []
    for i in range(19):
        issue = pd.Timestamp('2026-04-01T23:00Z') + pd.Timedelta(days=7 * i)
        for site, lat in [('A', 33.6), ('B', 33.9)]:
            for offset in range(1, 21):
                target = issue + pd.Timedelta(hours=offset)
                local = target.tz_convert('America/Chicago')
                rain = float((i + offset) % 5 == 0)
                rows.append({
                    'issued_at': issue, 'timestamp': target, 'location_id': site,
                    'local_date': str(local.date()), 'latitude': lat, 'longitude': -86.5,
                    'hour': local.hour, 'dow': local.dayofweek, 'month': local.month,
                    'is_weekend': int(local.dayofweek >= 5), 'is_holiday': 0,
                    'hour_sin': np.sin(local.hour*np.pi/12), 'hour_cos': np.cos(local.hour*np.pi/12),
                    'dow_sin': np.sin(local.dayofweek*2*np.pi/7),
                    'dow_cos': np.cos(local.dayofweek*2*np.pi/7),
                    'temperature_2m': 20 + i/4, 'precipitation': rain,
                    'cloud_cover': 30 + rain * 25, 'rain_prev24h': float(i%5),
                    'rain_prev72h': float(i%7), 'lead_day': 1,
                    'carwashes': float(max(0, 12 + ((i+offset)%8) - 6*rain + (site == 'B')*7)),
                    'inferred_zero': False,
                })
    examples = tmp_path / 'examples.csv'
    storm = tmp_path / 'storm.csv'
    baseline = tmp_path / 'baseline.csv'
    cfg = tmp_path / 'selected_configurations.json'
    pd.DataFrame(rows).to_csv(examples, index=False)
    raw = backtest.load_examples(examples)
    derived = bf.derive_features(raw, window_days=90, progress_every=0)
    derived.to_csv(baseline, index=False)
    bf.meta_path(baseline).write_text(json.dumps({
        'schema_version': 1, 'window_days': 90,
        'min_train_rows': 200, 'min_train_origins': 8,
        'examples_fingerprint': tuning.source_fingerprint(examples),
    }))
    ext = raw[bf.KEYS].copy()
    for name in sf.FEATURES:
        ext[name] = 1.0
    ext.to_csv(storm, index=False)
    cfg.write_text(json.dumps({'selected_models': {
        'xgboost': {'candidate_id': 'synthetic',
                    'parameter_overrides': {'n_estimators': 12, 'max_depth': 2}}
    }}))
    return examples, storm, baseline, cfg


def arguments(tmp_path):
    examples, storm, baseline, cfg = fixture_files(tmp_path)
    return argparse.Namespace(
        examples=str(examples), storm_features=str(storm),
        baseline_features=str(baseline), selected_configs=str(cfg),
        location='A', window_days=90,
        validation_from='2026-06-17', validation_to='2026-08-01',
        validation_origins=2, threads=2, output=str(tmp_path / 'out'),
    )


def test_model_configurations_are_local_or_global():
    assert lm.model_configuration('local_xgboost_storm') == (True, bi.REFERENCE)
    assert lm.model_configuration('global_xgboost_storm') == (False, bi.REFERENCE)
    assert lm.model_configuration('local_xgboost_storm_histmedian') == (
        True, 'xgboost_poisson_storm_histmedian')
    with pytest.raises(ValueError):
        lm.model_configuration('weather_median')


def test_end_to_end_resume_identical_targets_and_global_reference(tmp_path):
    args = arguments(tmp_path)
    pred, reports = lm.run(args)
    assert pred.model.nunique() == len(lm.VARIANTS)
    assert set(pred.location_id) == {'A'}
    assert pred.groupby('model').size().nunique() == 1
    assert len(pred[pred.model == lm.REFERENCE]) == 40
    assert (tmp_path / 'out' / 'gain_vs_global_by_regime.csv').exists()
    assert reports['overall_metrics.csv'].shape[0] == len(lm.VARIANTS)
    assert len(list((tmp_path / 'out' / 'fold_predictions').rglob('*.csv'))) == 12

    # Same cached train features/params and site test rows as established global XGBoost.
    data = bi.load_data(args.examples, args.storm_features, args.baseline_features, 90)
    issue = tuning.select_validation_issues(data, args.validation_from,
                                            args.validation_to, args.validation_origins)[0]
    train, test = backtest.fold_sets(data, issue, 90)
    test = test[test.location_id.eq('A')].copy().sort_values(lm.KEYS)
    expected = lm.train_predict(train, test, lm.REFERENCE,
                                bi.selected_xgboost_overrides(args.selected_configs), 2)
    got = pred[(pred.model == lm.REFERENCE) & (pred.issued_at == issue)].sort_values(lm.KEYS)
    np.testing.assert_allclose(got.prediction.to_numpy(), np.round(expected, 6), atol=1e-6)

    rerun, _ = lm.run(args)
    assert len(rerun) == len(pred)
    assert len(list((tmp_path / 'out' / 'fold_predictions').rglob('*.csv'))) == 12
    args.threads = 1
    with pytest.raises(ValueError, match='different run'):
        lm.run(args)


def test_local_training_excludes_other_store_and_enforces_folds(tmp_path):
    args = arguments(tmp_path)
    df = bi.load_data(args.examples, args.storm_features, args.baseline_features, 90)
    issue = tuning.select_validation_issues(df, args.validation_from,
                                            args.validation_to, args.validation_origins)[0]
    global_train, global_test = backtest.fold_sets(df, issue, 90)
    store = global_train[global_train.location_id.eq('A')].copy()
    test = global_test[global_test.location_id.eq('A')].copy()
    assert len(store) < len(global_train)
    assert (store.location_id == 'A').all()
    assert (store.timestamp < issue).all() and (store.issued_at < issue).all()
    overrides = bi.selected_xgboost_overrides(args.selected_configs)
    reference = lm.train_predict(store, test, 'local_xgboost_storm', overrides, 2)
    poisoned = global_train.copy()
    poisoned.loc[poisoned.location_id.eq('B'), 'carwashes'] = 99_999
    modified_store = poisoned[poisoned.location_id.eq('A')]
    check = lm.train_predict(modified_store, test, 'local_xgboost_storm', overrides, 2)
    np.testing.assert_allclose(reference, check, atol=0, rtol=0)


def test_unknown_store_and_unsafe_comparison_rejected(tmp_path):
    args = arguments(tmp_path)
    args.location = 'not-a-store'
    with pytest.raises(ValueError, match='not in examples'):
        lm.run(args)
    pred = pd.DataFrame([
        {'issued_at': '2026-06-01', 'location_id': 'A', 'timestamp': '2026-06-02',
         'carwashes': 10, 'absolute_error': 2, 'model': lm.REFERENCE}
    ])
    with pytest.raises(RuntimeError, match='missing or duplicated'):
        lm.paired_gains(pred, [])
