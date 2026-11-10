"""Independent lightweight checks for model_benchmark.py (no external data)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import backtest
import experiments
import model_benchmark as bm


def test_parse_models():
    assert bm.parse_models('boosted,xgboost') == ('boosted','xgboost')
    with pytest.raises(ValueError):
        bm.parse_models('catboost')
    with pytest.raises(ValueError):
        bm.parse_models('boosted,boosted')


def test_catboost_predict_returns_count_scale():
    model = bm.make_estimator('catboost', 2)
    x = np.arange(60, dtype=float).reshape(20, 3) / 20
    y = np.repeat([2., 16.], 10)
    model.fit(x, y)
    result = bm.response_prediction(model, 'catboost', x)
    assert (result > 0).all()
    assert np.allclose(result, model.predict(x, prediction_type='Exponent'))
    assert np.allclose(np.log(result), model.predict(x, prediction_type='RawFormulaVal'))


def test_missing_package_message(monkeypatch):
    native_import = bm.importlib.import_module
    def blocked(name):
        if name == 'xgboost':
            raise ImportError('testing a missing installation')
        return native_import(name)
    monkeypatch.setattr(bm.importlib, 'import_module', blocked)
    with pytest.raises(RuntimeError, match='pip install xgboost'):
        bm.require_libraries(('boosted', 'xgboost'))


def test_offline_folds_are_paired_and_match_old_boosted(tmp_path):
    # Generate two locations, 13 weekly issuances, and 48 targets per issuance.
    examples = []
    for i in range(13):
        issued = pd.Timestamp('2026-05-01T23:00:00Z') + pd.Timedelta(days=i * 7)
        for store, latitude in [('A', 33.6), ('B', 33.8)]:
            for h in range(24):
                target = issued + pd.Timedelta(hours=h + 1)
                day = target.tz_convert('America/Chicago')
                examples.append({
                    'issued_at': issued, 'timestamp': target,
                    'local_date': str(day.date()), 'location_id': store,
                    'latitude': latitude, 'longitude': -86.5, 'hour': day.hour,
                    'dow': day.dayofweek, 'month': day.month, 'is_weekend': int(day.dayofweek >= 5),
                    'is_holiday': 0, 'hour_sin': np.sin(day.hour*np.pi/12),
                    'hour_cos': np.cos(day.hour*np.pi/12),
                    'dow_sin': np.sin(day.dayofweek*2*np.pi/7),
                    'dow_cos': np.cos(day.dayofweek*2*np.pi/7),
                    'temperature_2m': 15+5*np.sin(i), 'precipitation': float((i+h)%7 == 0),
                    'cloud_cover': 50.0, 'rain_prev24h': float(i%3),
                    'rain_prev72h': float(i%5), 'lead_day': 1,
                    'carwashes': float(8+(i+h)%11), 'inferred_zero': False,
                })
    df = backtest.load_examples(_save_examples(tmp_path, pd.DataFrame(examples)))
    selected = ('baseline','boosted','lightgbm','xgboost','catboost')
    preds, reports, folds = bm.evaluate(df, window_days=90,
                                       test_origins=2, models=selected)
    assert len(folds) == 2
    assert set(preds.model.unique()) == set(selected)
    assert not preds.duplicated(['model']+bm.KEYS).any()
    assert (preds.prediction >= 0).all()
    assert np.isfinite(preds.prediction).all()
    assert len(preds[preds.model.eq('boosted')]) == len(preds[preds.model.eq('catboost')])
    # Compare old implementation on each of the exact same historical folds.
    enriched = experiments.add_season_features(df)
    for issue in bm.select_issuances(df, 2, None, None):
        train, test = backtest.fold_sets(enriched, issue, 90)
        old_model = experiments.boost_model(train, True)
        old_pred = old_model.predict(experiments.inputs(test, True))
        new_pred = preds.loc[(preds.model.eq('boosted')) & (preds.issued_at.eq(issue))]
        old = test[bm.KEYS].copy()
        old['pred'] = old_pred
        paired = new_pred.merge(old, on=bm.KEYS, validate='one_to_one')
        assert len(paired) == len(test)
        assert np.allclose(paired.prediction, paired.pred, atol=.0001)
    for name in ['overall_metrics.csv', 'by_weather_regime.csv', 'gain_vs_boosted.csv']:
        assert name in reports


def _save_examples(tmp_path, df):
    path = tmp_path/'examples.csv'
    df.to_csv(path, index=False)
    return path
