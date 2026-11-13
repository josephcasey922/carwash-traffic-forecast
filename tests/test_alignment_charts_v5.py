"""Chart smoke tests without launching a Streamlit server.

Load function definitions from the dashboard prefix into a minimal stub; this
checks plotted model names, chart series, and Plotly serialization.
"""
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from washcast_dashboard_core import prepare_predictions,matched_model_alignment, matched_alignment_by_period


@pytest.fixture(scope='module')
def ui_functions():
    dummy=types.ModuleType('streamlit')
    dummy.set_page_config=lambda **kwargs:None
    dummy.markdown=lambda *args,**kwargs:None
    dummy.cache_data=lambda **kwargs:(lambda f:f)
    old=sys.modules.get('streamlit')
    sys.modules['streamlit']=dummy
    try:
        source=(Path(__file__).resolve().parents[1]/'washcast_validation_dashboard.py').read_text()
        prefix=source.split("st.title('WashCast",1)[0]
        namespace={}
        exec(compile(prefix,'washcast_validation_dashboard.py','exec'),namespace)
        return namespace
    finally:
        if old is None:
            del sys.modules['streamlit']
        else:sys.modules['streamlit']=old


def models():
    rows=[]
    for model,pred,issued,lead in [
        ('lgbm_stable_recent',[18,50],'2026-11-02T00:00:00Z',1),
        ('weather_median',[19,45],'2026-11-02T00:00:00Z',1),
        ('weekly_sun__lgbm_stable_recent',[17,48],'2026-11-02T00:00:00Z',1),
    ]:
        for i,(hour,actual,guess) in enumerate(zip([8,9],[17,50],pred)):
            rows.append(dict(issued_at=issued,timestamp=f'2026-11-02T{hour+6:02d}:00:00Z',
                        location_id='AL003',local_date='2026-11-02',hour=hour,
                        lead_day=lead,carwashes=actual,prediction=guess,
                        model=model,inferred_zero=False))
    return prepare_predictions(pd.DataFrame(rows))


def test_alignment_chart_names_all_models(ui_functions):
    data=models()
    table=matched_model_alignment(data, ['lgbm_stable_recent','weather_median',
                                         'weekly_sun__lgbm_stable_recent'],
                                   match_target_hours=True)
    ns=ui_functions
    fig=ns['alignment_chart'](table,title='Test comparison')
    j=fig.to_plotly_json()
    assert len(j['data'])==1
    assert len(j['data'][0]['x'])==3
    assert 'lgbm_stable_recent · daily-updated Day 1' in j['data'][0]['x']
    assert 'weather_median · daily-updated Day 1' in j['data'][0]['x']
    assert 'lgbm_stable_recent · Sunday 6 PM fixed weekly' in j['data'][0]['x']
    dayplot=ns['plot_hours'](data,'lgbm_stable_recent','weather_median',weekly_model='weekly_sun__lgbm_stable_recent')
    labels=[trace.name for trace in dayplot.data]
    assert 'lgbm_stable_recent · daily-updated Day 1' in labels
    assert 'weather_median · daily-updated Day 1' in labels
    assert 'lgbm_stable_recent · Sunday 6 PM fixed weekly' in labels


def test_weekly_chart_three_forecasts_and_trend(ui_functions):
    data=models()
    ns=ui_functions
    plot=ns['weekly_small_multiples'](data,'weekly_sun__lgbm_stable_recent',
              pd.Timestamp('2026-11-02'),comparison='lgbm_stable_recent',extra_comparison='weather_median')
    names={x.name for x in plot.data}
    assert 'lgbm_stable_recent · Sunday 6 PM fixed weekly' in names
    assert 'lgbm_stable_recent · daily-updated Day 1' in names
    assert 'weather_median · daily-updated Day 1' in names
    trend=matched_alignment_by_period(data,['weekly_sun__lgbm_stable_recent',
              'lgbm_stable_recent','weather_median'],period='day',match_target_hours=True)
    fig=ns['alignment_period_chart'](trend,title='Day trend')
    assert len(fig.data)==3
    assert {x.name for x in fig.data}==names-{'Actual washes'}
