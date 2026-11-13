"""Causality and count-calibration tests; unittest, no external test plugin needed."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import washcast_adaptation_experiment as adaptive


class TestAdaptation(unittest.TestCase):
    @staticmethod
    def examples(days=27, bias=-4., extra_store=False):
        starts=pd.date_range('2026-05-01',periods=days)
        rows=[]
        sites=['AL003','AL010'] if extra_store else ['AL003']
        for target in starts:
            for site in sites:
                for hour in range(9,19) if target.dayofweek==6 else range(8,19):
                    pred=30. if site=='AL003' else 15.
                    actual=pred - bias if site=='AL003' else pred
                    rows.append({'issued_at':(target-pd.Timedelta(days=1)).tz_localize('America/Chicago') +pd.Timedelta(hours=18),
                                 'location_id':site,'local_date':target,'hour':hour,
                                 'carwashes':actual,'prediction':pred,'inferred_zero':False})
        return pd.DataFrame(rows)

    def test_buckets_round_half_up(self):
        np.testing.assert_array_equal(adaptive.buckets(np.array([0,17.49,17.5,40.4,40.5,60.5,90])),
                                      np.array([0,0,1,1,2,3,3]))

    def test_only_prior_complete_target_days(self):
        d=self.examples()
        d['error']=d.prediction-d.carwashes
        day=adaptive.daily_history(d)
        target=pd.Timestamp('2026-05-14')
        val,record=adaptive.store_offset(day,target,'AL003',span=14,min_days=7,
                    sign_requirement=.65,min_washes=.75,magnitude_fraction=.05,
                    strength=.6,cap=4.)
        self.assertGreater(val,0)
        self.assertTrue(record['triggered'])
        after=d.copy()
        # Changing D-1 (unavailable at previous-day 6pm) cannot affect target D.
        after.loc[after.local_date.eq(target-pd.Timedelta(days=1)),'carwashes']=100000
        after['error']=after.prediction-after.carwashes
        later=adaptive.daily_history(after)
        other,_=adaptive.store_offset(later,target,'AL003',span=14,min_days=7,
                    sign_requirement=.65,min_washes=.75,magnitude_fraction=.05,
                    strength=.6,cap=4.)
        self.assertEqual(val,other)

    def test_no_lookahead_at_all_with_future_modified(self):
        x=self.examples(extra_store=True)
        x['error']=x.prediction-x.carwashes
        x['absolute_error']=x.error.abs()
        x['volume_band']=adaptive.volume_band(x.prediction)
        before,_=adaptive.apply_candidates(x)
        altered=x.copy()
        altered.loc[altered.local_date>=pd.Timestamp('2026-05-17'),'carwashes']=90000
        altered['error']=altered.prediction-altered.carwashes
        altered['absolute_error']=altered.error.abs()
        after,_=adaptive.apply_candidates(altered)
        pred_a=before[before.local_date<pd.Timestamp('2026-05-17')].adjusted_prediction
        pred_b=after[after.local_date<pd.Timestamp('2026-05-17')].adjusted_prediction
        np.testing.assert_allclose(pred_a.to_numpy(),pred_b.to_numpy())

    def test_insufficient_history_no_changes(self):
        d=self.examples(days=4)
        d['error']=d.prediction-d.carwashes
        d['volume_band']=adaptive.volume_band(d.prediction)
        y,h=adaptive.apply_candidates(d)
        np.testing.assert_allclose(y.adjusted_prediction,y.prediction)
        self.assertEqual(int(h.fast_gate.sum()),0)

    def test_future_incomplete_zeros_not_used_for_training(self):
        d=self.examples()
        d['error']=d.prediction-d.carwashes
        d['volume_band']=adaptive.volume_band(d.prediction)
        d.loc[(d.local_date==pd.Timestamp('2026-05-12'))&(d.hour.isin([8,9,10])),'inferred_zero']=True
        daily=adaptive.daily_history(d)
        self.assertFalse(bool(daily.loc[daily.local_date==pd.Timestamp('2026-05-12'),'eligible'].iloc[0]))

    def test_file_loading_and_strict_next_day(self):
        d=self.examples(days=2)
        d['lead_day']=1
        d['model']='lgbm_stable_recent'
        with tempfile.TemporaryDirectory() as td:
            fp=Path(td)/'data.csv'
            d.to_csv(fp,index=False)
            parsed=adaptive.read_predictions([fp],'lgbm_stable_recent')
            self.assertEqual(len(parsed),len(d))
            bad=d.copy()
            bad['issued_at']=bad.issued_at+pd.Timedelta(days=1)
            bad.to_csv(fp,index=False)
            with self.assertRaises(ValueError):
                adaptive.read_predictions([fp],'lgbm_stable_recent')

    def test_end_to_end(self):
        d=self.examples(days=24,extra_store=True)
        d['lead_day']=1
        d['model']='lgbm_stable_recent'
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'data.csv';d.to_csv(source,index=False)
            reports=adaptive.run(paths=[source],model='lgbm_stable_recent',output=Path(td)/'out')
            self.assertEqual(len(reports['overall_day1.csv']),len(adaptive.CANDIDATES))
            self.assertTrue((Path(td)/'out'/'by_store_day1.csv').exists())
            self.assertEqual(int(reports['overall_day1.csv'].iloc[0].hours),len(d))


if __name__=='__main__':
    unittest.main()
