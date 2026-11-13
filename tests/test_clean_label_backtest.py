import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from clean_label_backtest import synchronized_gap_days, training_filter, attach_tags, metrics, make_quality_tags


def test_training_filter_keeps_original_and_removes_only_targeted():
    x = pd.DataFrame({'flagged_store_day':[False,True,True,False],
                      'inferred_zero':[False,False,True,True],
                      'synchronized_gap_day':[False,False,False,True]})
    assert training_filter(x,'original').tolist() == [True]*4
    assert training_filter(x,'drop_flagged_zeros').tolist() == [True,True,False,True]
    assert training_filter(x,'drop_flagged_days').tolist() == [True,False,False,True]
    assert training_filter(x,'drop_flagged_and_network_days').tolist() == [True,False,False,False]
    with pytest.raises(ValueError):training_filter(x,'invalid')


def test_sync_gap_detects_short_network_closure_even_if_each_store_below_four():
    rows=[]
    for store in range(15):
        for hour in range(8,19):
            rows.append(dict(local_date='2026-07-04',hour=hour,
                location_id=f'AL{store:03d}',inferred_zero=hour in (17,18)))
    assert '2026-07-04' in synchronized_gap_days(pd.DataFrame(rows))
    # A single unusual hour is not enough to trigger a day-level warning.
    for row in rows:
        if row['hour']==18:row['inferred_zero']=False
    assert '2026-07-04' not in synchronized_gap_days(pd.DataFrame(rows))


def test_attach_tags_correct_many_to_one_and_provenance():
    labels=pd.DataFrame({'location_id':['AL002','AL002'],
        'local_date':['2026-05-01','2026-05-01'],
        'carwashes':[5,0],'inferred_zero':['False','True']})
    tags=pd.DataFrame({'location_id':['AL002'],'local_date':['2026-05-01'],
        'quality_tier':['B_1to3_inferred'],'flagged_store_day':[False],
        'synchronized_gap_day':[False]})
    x=attach_tags(labels,tags)
    assert x.inferred_zero.tolist()==[False,True]
    assert x.quality_tier.nunique()==1
    with pytest.raises(ValueError):attach_tags(labels.assign(carwashes=[0,0]),tags)


def test_metrics_bucket_boundaries():
    x=pd.DataFrame({'prediction':[17.0,18.0,40.0,41.0,60.0,61.0],
                    'carwashes':[18,17,41,40,61,60]})
    z=metrics(x)
    assert z['hours']==6
    assert z['bucket_agreement_percent']==0
    assert z['bucket_under_percent']==50


def test_quality_dates_are_strings_and_synchronous_boolean():
    d=pd.DataFrame({'location_id':['AL002'],'local_date':pd.to_datetime(['2026-07-04']),
        'quality_tier':['B_1to3_inferred'],'inferred_hours':[3],
        'missing_scheduled_hours':[0]})
    tags=make_quality_tags(None,d,{'2026-07-04'})
    assert bool(tags.synchronized_gap_day.iloc[0])
    assert not bool(tags.flagged_store_day.iloc[0])
