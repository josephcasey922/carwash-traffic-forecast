import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from washcast_rollout_audit import bucket, load_residuals, measurements, run_causal_monitor


class RolloutAuditTests(unittest.TestCase):
    def test_bucket_edges_half_up(self):
        a = np.array([0, 17.49, 17.50, 18, 40.49, 40.50, 60.49, 60.50, 61, 100.])
        np.testing.assert_array_equal(bucket(a), [0, 0, 1, 1, 1, 2, 2, 3, 3, 3])

    def make_data(self):
        dates = pd.date_range('2026-08-01', periods=38, freq='D')
        d = pd.DataFrame([{'location_id': 'AL003', 'local_date': date,
             'hour': 10, 'carwashes': 20 if i < 24 else 8,
             'prediction': 10, 'inferred_zero': False}
             for i,date in enumerate(dates)])
        d['error'] = d.prediction - d.carwashes
        d['absolute_error'] = d.error.abs()
        d['actual_bucket'] = bucket(d.carwashes)
        d['forecast_bucket'] = bucket(d.prediction)
        d['bucket_error'] = d.forecast_bucket - d.actual_bucket
        return d

    def test_next_day_label_unavailable(self):
        d = self.make_data()
        f, _ = run_causal_monitor(d, set())
        date = pd.Timestamp('2026-08-26')
        current = f.loc[f.target_date==date.date()].iloc[0]
        # Changing same-day and yesterday actuals cannot change this forecast-time alert.
        d2 = d.copy()
        d2.loc[d2.local_date>=date-pd.Timedelta(days=1), 'carwashes'] = 999
        d2['error'] = d2.prediction - d2.carwashes
        f2, _ = run_causal_monitor(d2, set())
        updated = f2.loc[f2.target_date==date.date()].iloc[0]
        self.assertEqual(current.alert, updated.alert)
        self.assertAlmostEqual(current.exploratory_offset, updated.exploratory_offset)
        self.assertAlmostEqual(current.historical_median_hourly_bias, updated.historical_median_hourly_bias)

    def test_no_correction_without_history(self):
        f, _ = run_causal_monitor(self.make_data(), set())
        self.assertFalse(f.iloc[0].alert)
        self.assertEqual(f.iloc[0].exploratory_offset, 0.)
        self.assertTrue((f.head(10).exploratory_offset==0).all())

    def test_duplicate_input_rejected(self):
        d = self.make_data()
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'d.csv'
            pd.concat([d.iloc[:1],d.iloc[:1]]).to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError, 'Nonunique'):
                load_residuals(p)

    def test_summary_valid(self):
        d=self.make_data()
        m=measurements(d)
        self.assertEqual(m['observations'],38)
        self.assertTrue(0<=m['bucket_agreement_percent']<=100)


if __name__ == '__main__':
    unittest.main()
