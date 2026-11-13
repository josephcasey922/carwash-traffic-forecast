import pandas as pd
import numpy as np
import pytest

from washcast_label_audit import longest_run, load_unique_labels, make_days, missing_store_days, raw_crosscheck


def fixture_examples(tmp_path):
    # Monday: 11 hours, with one inferred zero, repeated as forecast targets.
    dates=['2026-05-04']*11
    hrs=list(range(8,19))
    h=pd.DataFrame({'location_id':['AL002']*11,
      'timestamp':pd.to_datetime([f'2026-05-04 {x:02d}:00:00+00:00' for x in hrs],utc=True),
      'local_date':dates,'hour':hrs,
      'carwashes':[0 if x == 10 else 5 for x in hrs],
      'inferred_zero':[x==10 for x in hrs]})
    # Monday example, two separate forecast issue copies.
    example=pd.concat([h,h],ignore_index=True)
    fp=tmp_path/'examples.csv';example.to_csv(fp,index=False)
    return h,fp


def test_longest_run():
    assert longest_run(np.array([8,9,11,12,13,18]))==3
    assert longest_run(np.array([],dtype=int))==0
    assert longest_run(np.array([8,10,12]))==1


def test_history_dedup_and_day_quality(tmp_path):
    h,fp=fixture_examples(tmp_path)
    hist,info,_=load_unique_labels(fp)
    assert info['all_example_rows']==22
    assert info['unique_store_target_hours']==11
    assert info['unique_inferred_zero_hours']==1
    day=make_days(hist)
    assert day.iloc[0].quality_tier=='B_1to3_inferred'
    assert day.iloc[0].longest_inferred_run==1
    assert len(missing_store_days(day))==0


def test_source_crosscheck(tmp_path):
    h,fp=fixture_examples(tmp_path)
    raw=h[~h.inferred_zero].rename(columns={'timestamp':'hour','location_id':'location_code_id','carwashes':'num_carwashes'})
    raw=raw[['hour','location_code_id','num_carwashes']]
    rawpath=tmp_path/'raw.csv';raw.to_csv(rawpath,index=False)
    hist,_,_=load_unique_labels(fp)
    errors,metrics=raw_crosscheck(rawpath,hist)
    assert errors.empty
    assert metrics['raw_source_rows']==10
    assert metrics['labels_inferred_but_in_raw']==0


def test_contradictory_target_fails(tmp_path):
    _,fp=fixture_examples(tmp_path)
    bad=pd.read_csv(fp)
    bad.loc[11,'carwashes']=999
    bad.to_csv(fp,index=False)
    with pytest.raises(ValueError,match='Contradictory'):
        load_unique_labels(fp)
