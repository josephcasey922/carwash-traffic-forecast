"""Persistence and incremental learning regression tests."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import asof


def test_saved_keys_are_unique_and_keep_partially_labeled_origins(tmp_path):
    base = pd.DataFrame({
        'issued_at': ['2026-11-01T23:00:00Z'] * 2,
        'location_id': ['101','101'],
        'timestamp': ['2026-11-02T15:00:00Z','2026-11-02T16:00:00Z'],
    })
    asof.labeled_keys(base).to_csv(tmp_path/'model_training_keys.csv',index=False)
    old = asof.saved_labeled_keys(tmp_path)
    assert len(old)==2
    new=base.copy()
    new.loc[1,'timestamp']='2026-11-03T15:00:00Z'
    unseen = asof.labeled_keys(new).merge(old,on=asof.KEY_COLS,how='left',indicator=True)
    assert unseen['_merge'].tolist()==['both','left_only']


def test_old_artifact_dependency_warns_with_clear_error(tmp_path):
    (tmp_path/'models').mkdir()
    with pytest.raises(RuntimeError,match='Loading across versions is unsupported'):
        asof.load_saved_model(tmp_path,'mlp',{'runtime_versions':{'scikit_learn':'0.0.0'}})


def test_old_example_model_mismatch_reports_actionable_error(tmp_path):
    (tmp_path/'models').mkdir()
    (tmp_path/'models'/'mlp.joblib').write_bytes(b'invalid')
    with pytest.raises(Exception):
        asof.load_saved_model(tmp_path,'mlp',{'runtime_versions':{'scikit_learn':asof.sklearn.__version__}})


def test_runtime_versions_contains_scikit_and_python():
    assert 'scikit_learn' in asof.runtime_versions()
    assert 'python' in asof.runtime_versions()
