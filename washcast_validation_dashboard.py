"""Read-only, local historical WashCast forecast dashboard.

Install: python -m pip install streamlit plotly pandas openpyxl
Run:     python -m streamlit run washcast_validation_dashboard.py

Forecasts stay on the computer running Streamlit; no web/API/training calls.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from washcast_dashboard_core import (
    RANGES, WEEKLY_PREFIX, employee_counts, combine_predictions, daily_metrics, daily_totals,
    matched_model_alignment, matched_alignment_by_period,
    discover_prediction_csvs, load_location_names, metrics, month_summary,
    paired_predictions, matched_daily_weekly, sunday_weekly_rows,
    prepare_predictions, prepare_upcoming_weekly, quality_cohort, store_scorecard, week_start,
)

st.set_page_config(page_title='WashCast | Forecast Explorer',page_icon='🚘',layout='wide')
BLUE='#3775CD'; ORANGE='#F2994A'; GRAY='#63758D'; RED='#C34A52'; GREEN='#158A74'; PURPLE='#9459B9'

st.markdown('''<style>
.block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
[data-testid="stMetricValue"]{font-weight:650;letter-spacing:-.025em}
[data-testid="stSidebar"]{border-right:1px solid #DEE4EC;}
</style>''',unsafe_allow_html=True)


def clean_num(x,unit='',dec=2):
    if x is None or not np.isfinite(x):return '—'
    return f'{x:,.{dec}f}{unit}'




def model_label(model: str, *, lead_day: int = 1) -> str:
    """Always identify which forecast generated an operational score."""
    if str(model).startswith(WEEKLY_PREFIX):
        return f'{str(model)[len(WEEKLY_PREFIX):]} · Sunday 6 PM fixed weekly'
    return f'{model} · daily-updated Day {lead_day}'


def alignment_chart(table: pd.DataFrame, *, title: str, lead_day: int = 1):
    fig=go.Figure()
    if not table.empty:
        labels=[model_label(m,lead_day=lead_day) for m in table.model]
        colors=[PURPLE if str(m).startswith(WEEKLY_PREFIX) else
                BLUE if i==0 else ORANGE if i==1 else GREEN
                for i,m in enumerate(table.model)]
        fig.add_trace(go.Bar(x=labels,y=table.forecast_labor_alignment_percent,
            marker_color=colors,
            text=[f'{x:.2f}%' for x in table.forecast_labor_alignment_percent],
            textposition='outside',cliponaxis=False,
            customdata=table[['hours','recommended_employee_hours',
                              'absolute_employee_hour_difference']].to_numpy(),
            hovertemplate='%{x}<br>Alignment %{y:.2f}%<br>Matched hours %{customdata[0]:,.0f}'
                '<br>Recommended employee-hours %{customdata[1]:,.0f}'
                '<br>Absolute employee-hour difference %{customdata[2]:,.0f}<extra></extra>'))
    fig.update_yaxes(title='Forecast Labor Alignment (%)')
    fig.update_layout(title=title,showlegend=False)
    return style_fig(fig,330)


def alignment_period_chart(table: pd.DataFrame, *, title: str, lead_day: int = 1):
    fig=go.Figure()
    if not table.empty:
        for i,(model,g) in enumerate(table.groupby('model',sort=False,observed=True)):
            color=PURPLE if str(model).startswith(WEEKLY_PREFIX) else BLUE if i==0 else ORANGE if i==1 else GREEN
            fig.add_trace(go.Scatter(x=g.period,y=g.forecast_labor_alignment_percent,
                    mode='lines+markers',name=model_label(model,lead_day=lead_day),
                    line=dict(color=color,width=2.8),
                    customdata=g[['hours','absolute_employee_hour_difference',
                                  'recommended_employee_hours']].to_numpy(),
                    hovertemplate='%{x}<br>Alignment %{y:.2f}%<br>Matched hours %{customdata[0]:.0f}'
                      '<br>Absolute difference %{customdata[1]:.0f}'
                      '<br>Recommended employee-hours %{customdata[2]:.0f}<extra></extra>'))
    fig.update_layout(title=title)
    fig.update_yaxes(title='Forecast Labor Alignment (%)')
    fig.update_xaxes(title='Target date / period')
    return style_fig(fig,360)


def show_matched_alignment(frame: pd.DataFrame, models: list[str], *,
                           title: str, lead_day: int = 1, period: str | None = None):
    """Model-labelled alignment, scored only on the common actual target hours."""
    present=[m for m in dict.fromkeys(models) if m and frame.model.eq(m).any()]
    if len(present)<1:return
    includes_weekly=any(str(m).startswith(WEEKLY_PREFIX) for m in present)
    try:
        table=matched_model_alignment(frame,present,match_target_hours=includes_weekly)
        if table.empty:
            st.info('No identical store-hours across the selected models for alignment comparison.')
            return
        table=table.copy()
        table['Forecast / model']=[model_label(m,lead_day=lead_day) for m in table.model]
        st.markdown(f'**{title}**')
        st.caption(f'Every model is compared on the same {int(table.hours.iloc[0]):,} observed store-hours. '
            'Alignment = 100 × (1 − sum of absolute employee-hour differences ÷ sum of '
            'actual-wash recommended employee-hours); it is NOT an average of daily scores.')
        values=st.columns(len(table))
        for col,(_,row) in zip(values,table.iterrows()):
            col.metric(row['Forecast / model'],clean_num(row.forecast_labor_alignment_percent,'%'),
                help=f'Absolute employee-hour difference: {int(row.absolute_employee_hour_difference)} '
                     f'/ actual recommended employee-hours: {int(row.recommended_employee_hours)}')
        st.plotly_chart(alignment_chart(table,title='Matched-hour alignment by forecast',lead_day=lead_day),
                        use_container_width=True)
        st.dataframe(table[['Forecast / model','hours','forecast_labor_alignment_percent',
                            'bucket_agreement','recommended_employee_hours',
                            'absolute_employee_hour_difference',
                            'under_recommended_employee_hours','over_recommended_employee_hours']].rename(columns={
            'hours':'Matched hours','forecast_labor_alignment_percent':'Alignment (%)',
            'bucket_agreement':'Correct bucket (%)','recommended_employee_hours':'Recommended employee-hours',
            'absolute_employee_hour_difference':'Absolute employee-hour error',
            'under_recommended_employee_hours':'Under-recommended employee-hours',
            'over_recommended_employee_hours':'Over-recommended employee-hours'}),
            hide_index=True,use_container_width=True)
        if period is not None:
            trend=matched_alignment_by_period(frame,present,period=period,
                                                match_target_hours=includes_weekly)
            st.plotly_chart(alignment_period_chart(trend,
                title=f'{period.title()} alignment by forecast (matched store-hours)',
                lead_day=lead_day),use_container_width=True)
            # Keep all original model identifiers in downloadable results.
            st.download_button('Download matched model alignment',trend.to_csv(index=False),
                file_name=f'washcast_matched_alignment_{period}.csv',mime='text/csv',
                key=f'export_alignment_{period}_{title}')
    except ValueError as exc:
        st.warning(f'Alignment comparison unavailable: {exc}')


def fmt_hour(h:int)->str:
    return f'{(int(h)-1)%12+1} {"AM" if h<12 else "PM"}'


def style_fig(fig,height=400):
    fig.update_layout(template='plotly_white',margin=dict(l=16,r=12,t=48,b=20),
                      height=height,legend=dict(orientation='h',y=1.13,x=0),
                      hovermode='x unified',font=dict(family='Arial'))
    return fig


def add_bucket_lines(fig):
    for y in (17.5,40.5,60.5):
        fig.add_hline(y=y,line=dict(color='#B1B9C4',width=1,dash='dot'),
                      annotation_text=f'{int(y+.5)}',annotation_position='top right',
                      annotation_font_color='#708096')


def plot_hours(sub,model,comparison=None,buckets=False,weekly_model=None,lead_day=1):
    fig=go.Figure()
    actual=sub.loc[sub.model.eq(model)].sort_values('hour')
    if actual.empty:return fig
    fig.add_trace(go.Scatter(x=actual.hour,y=actual.carwashes,name='Actual',mode='lines+markers',
                             line=dict(color=GRAY,width=3),marker=dict(size=8)))
    for candidate,color in [(model,BLUE),(comparison,ORANGE)]:
        if not candidate:continue
        df=sub.loc[sub.model.eq(candidate)].sort_values('hour')
        if df.empty:continue
        fig.add_trace(go.Scatter(x=df.hour,y=df.prediction,name=model_label(candidate,lead_day=lead_day),
                                 mode='lines+markers',line=dict(color=color,width=2.6),
                                 marker=dict(size=7)))
    if weekly_model:
        q=sub.loc[sub.model.eq(weekly_model)].sort_values('hour')
        if not q.empty:
            fig.add_trace(go.Scatter(x=q.hour,y=q.prediction,name=model_label(weekly_model),
                     mode='lines+markers',line=dict(color=PURPLE,width=2.5,dash='dash'),
                     marker=dict(symbol='diamond',size=7),
                     customdata=np.stack([q.lead_day.to_numpy()],axis=-1),
                     hovertemplate='Hour %{x}:00<br>Sunday weekly forecast %{y:.1f}<br>Sunday-issued lead day %{customdata[0]}<extra></extra>'))
    zeros=actual.loc[actual.inferred_zero.eq(True)]
    if len(zeros):
        fig.add_trace(go.Scatter(x=zeros.hour,y=zeros.carwashes,name='Inferred zero (unverified)',
                                 mode='markers',marker=dict(color=RED,symbol='x',size=14,line=dict(width=2))))
    if buckets:add_bucket_lines(fig)
    fig.update_xaxes(tickvals=list(range(8,19)),ticktext=[fmt_hour(h) for h in range(8,19)],
                     title='Local operating hour')
    fig.update_yaxes(title='Car washes / hour',rangemode='tozero')
    return style_fig(fig,460)


def hour_error_chart(df):
    fig=go.Figure()
    df=df.sort_values('hour')
    fig.add_bar(x=df.hour,y=df.signed_error,marker_color=np.where(df.signed_error<0,RED,GREEN),
                customdata=np.stack([df.carwashes.to_numpy(),df.prediction.to_numpy()],axis=-1),
                hovertemplate='Hour %{x}:00<br>Error %{y:+.2f}<br>Actual %{customdata[0]:.0f}<br>Pred %{customdata[1]:.2f}<extra></extra>')
    fig.add_hline(y=0,line_width=1,line_color='#789')
    fig.update_xaxes(tickvals=list(range(8,19)),ticktext=[fmt_hour(h) for h in range(8,19)])
    fig.update_yaxes(title='Predicted − actual (washes)')
    return style_fig(fig,230)


def weekly_small_multiples(frame,model,week,comparison=None,buckets=False,weekly_model=None,extra_comparison=None,lead_day=1):
    fig=make_subplots(rows=2,cols=4,subplot_titles=[
        (week+pd.Timedelta(days=i)).strftime('%a %b %d')
        for i in range(7)],vertical_spacing=.17,horizontal_spacing=.055)
    plotted=set()
    for i in range(7):
        day=week+pd.Timedelta(days=i)
        group=frame.loc[frame.local_date.eq(day)]
        if group.empty:continue
        model_color=PURPLE if str(model).startswith(WEEKLY_PREFIX) else BLUE
        comparison_color=BLUE if str(model).startswith(WEEKLY_PREFIX) else ORANGE
        for name,color,field in [('Actual',GRAY,'carwashes'),(model,model_color,'prediction'),
                                 (comparison,comparison_color,'prediction'),
                                 (extra_comparison,ORANGE,'prediction'),
                                 (weekly_model,PURPLE,'prediction')]:
            if not name:continue
            q=group.loc[group.model.eq(model if name=='Actual' else name)].sort_values('hour')
            if q.empty:continue
            fig.add_trace(go.Scatter(x=q.hour,y=q[field],mode='lines+markers',name=(model_label(name,lead_day=lead_day) if name!='Actual' else 'Actual washes'),
                                     legendgroup=name,showlegend=name not in plotted,
                                     line=dict(color=color,width=2,
                                               dash='dash' if str(name).startswith(WEEKLY_PREFIX) else 'solid'),
                                     marker=dict(size=4)),
                          row=i//4+1,col=i%4+1)
            plotted.add(name)
        fig.update_xaxes(tickvals=[8,10,12,14,16,18],ticktext=['8','10','12','14','16','18'],
                         row=i//4+1,col=i%4+1)
        fig.update_yaxes(rangemode='tozero',row=i//4+1,col=i%4+1)
    fig.update_layout(title='Hourly wash traffic across the selected week')
    return style_fig(fig,640)


def weekly_signed_heatmap(frame,model,week):
    q=frame.loc[frame.model.eq(model)].copy()
    if q.empty:return go.Figure()
    q['weekday']=q.local_date.dt.dayofweek
    mat=q.pivot_table(index='weekday',columns='hour',values='signed_error',aggfunc='mean')
    mat=mat.reindex(index=range(7),columns=range(8,19))
    vmax=float(max(np.nanmax(np.abs(mat.to_numpy(float))),1)) if mat.notna().any().any() else 1
    fig=go.Figure(go.Heatmap(z=mat.to_numpy(),x=[fmt_hour(h) for h in mat.columns],
                   y=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],
                   zmin=-vmax,zmax=vmax,zmid=0,colorscale='RdBu',reversescale=True,
                   colorbar=dict(title='Pred − actual'),hoverongaps=False,
                   hovertemplate='%{y} %{x}<br>Signed error %{z:.1f} washes<extra></extra>'))
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(title='Where errors occur (signed washes/hour)')
    return style_fig(fig,310)


def daily_traffic_chart(ds,model,comparison=None):
    fig=go.Figure()
    actual=ds.loc[ds.model.eq(model)&ds.complete].sort_values('local_date')
    fig.add_trace(go.Scatter(x=actual.local_date,y=actual.actual,name='Actual daily total',
                  mode='lines+markers',line=dict(color=GRAY,width=2),marker=dict(size=4)))
    for name,color in [(model,BLUE),(comparison,ORANGE)]:
        if not name:continue
        q=ds.loc[ds.model.eq(name)&ds.complete].sort_values('local_date')
        fig.add_trace(go.Scatter(x=q.local_date,y=q.predicted,name=f'{name} daily total',
                      mode='lines',line=dict(color=color,width=2)))
    fig.update_xaxes(title='Target date')
    fig.update_yaxes(title='Car washes / complete day',rangemode='tozero')
    return style_fig(fig,350)


def monthly_chart(frame,lead_day=1):
    monthly=month_summary(frame)
    fig=make_subplots(rows=1,cols=3,subplot_titles=['Hourly MAE (washes)','Hourly WAPE (%)',
                                                     'Forecast Labor Alignment (%)'])
    for model,g in monthly.groupby('model',observed=True):
        g=g.sort_values('month')
        color=PURPLE if str(model).startswith(WEEKLY_PREFIX) else BLUE if model==primary else ORANGE
        fig.add_trace(go.Scatter(x=g.month,y=g.mae,name=model_label(model,lead_day=lead_day),mode='lines+markers',
                      legendgroup=model,line=dict(color=color,width=3)),row=1,col=1)
        fig.add_trace(go.Scatter(x=g.month,y=g.wape,name=model,mode='lines+markers',
                      legendgroup=model,showlegend=False,line=dict(color=color,width=3)),row=1,col=2)
        fig.add_trace(go.Scatter(x=g.month,y=g.forecast_labor_alignment_percent,name=model,
                      mode='lines+markers',legendgroup=model,showlegend=False,
                      line=dict(color=color,width=3)),row=1,col=3)
    fig.update_yaxes(title_text='Percent',row=1,col=3)
    return style_fig(fig,355)


def upcoming_seven_days_chart(future, monday, buckets=False):
    fig=make_subplots(rows=2,cols=4,subplot_titles=[
        (monday+pd.Timedelta(days=i)).strftime('%a %b %d') for i in range(7)],
        vertical_spacing=.17,horizontal_spacing=.055)
    for idx in range(7):
        day=monday+pd.Timedelta(days=idx)
        g=future.loc[future.local_date.eq(day)].sort_values('hour')
        if not g.empty:
            fig.add_trace(go.Scatter(x=g.hour,y=g.prediction,mode='lines+markers',
                 name='Sunday weekly forecast',showlegend=idx==0,
                 line=dict(color=PURPLE,width=2.5),marker=dict(size=5)),
                 row=idx//4+1,col=idx%4+1)
            if buckets:
                for threshold in [17.5,40.5,60.5]:
                    fig.add_hline(y=threshold,line=dict(color='#B1B9C4',width=1,dash='dot'),
                                  row=idx//4+1,col=idx%4+1)
        fig.update_xaxes(tickvals=[8,10,12,14,16,18],row=idx//4+1,col=idx%4+1)
        fig.update_yaxes(rangemode='tozero',row=idx//4+1,col=idx%4+1)
    fig.update_layout(title='Unscored Sunday-issued forecast for the full workweek')
    return style_fig(fig,650)


@st.cache_data(show_spinner=False)
def cached_read(path_str:str,stamp:int,size:int):
    return prepare_predictions(pd.read_csv(path_str),source=str(Path(path_str).parent.name))


@st.cache_data(show_spinner=False)
def cached_upcoming(path_str:str,stamp:int,size:int):
    return prepare_upcoming_weekly(pd.read_csv(path_str,dtype={'location_id':str}))


@st.cache_data(show_spinner=False)
def cached_uploaded(blob:bytes,name:str):
    import io
    return prepare_predictions(pd.read_csv(io.BytesIO(blob)),source=f'Uploaded: {name}')


st.title('WashCast  /  Forecast explorer')
st.caption('Historical as-issued wash forecasts · Local-only viewer · Model-labelled labor alignment for each day and week · Wash-count accuracy first')

with st.sidebar:
    st.markdown('### Prediction sources')
    root_str=st.text_input('Project root folder',value=str(Path.cwd()),help='Folder containing demand_ablation_*, clean_label_*, and location_id_mapping.csv')
    root=Path(root_str).expanduser()
    paths=discover_prediction_csvs(root)
    path_labels=[str(p.relative_to(root)) if root.is_dir() else str(p) for p in paths]
    # Only default-select standard ablation runs. 'adaptation_may_july_daily'
    # contains the same substring but is a separate, optional comparison.
    default=[s for s,p in zip(path_labels,paths) if
             p.parent.name in ('demand_ablation_may_july_daily','demand_ablation_aug_sep_all_origins')
             and p.name == 'all_predictions.csv']
    chosen=st.multiselect('Saved experiment CSVs',options=path_labels,default=default or path_labels[:1],
                          help='Select runs with different date ranges. Overlapping predictions must match exactly.')
    manual=st.text_area('Additional local CSV paths (optional)',height=65,
                        help='One absolute or relative path per line, e.g. demand_ablation_oct_2026/all_predictions.csv')
    uploads=st.file_uploader('Or select prediction CSVs from your computer',type=['csv'],accept_multiple_files=True)
    future_files=sorted(root.glob('sunday_weekly_live/upcoming_weekly_forecast*.csv')) if root.is_dir() else []
    future_options=['None']+[str(p.relative_to(root)) for p in future_files]
    future_choice=st.selectbox('Upcoming Sunday forecast (unscored)',future_options,
                               index=len(future_options)-1 if len(future_options)>1 else 0,
                               help='Select a separately issued future week. Never scored until its actual washes exist.')
    st.divider()
    st.caption('The app never trains models, fetches weather, or changes historical observations. Do not publish a dashboard containing private traffic data.')

try:
    frames=[]
    chosen_paths=[paths[path_labels.index(v)] for v in chosen]
    for line in manual.splitlines():
        q=line.strip().strip('"')
        if not q:continue
        f=Path(q).expanduser()
        if not f.is_absolute():f=root/f
        if not f.is_file():raise ValueError(f'File does not exist: {f}')
        if f not in chosen_paths:chosen_paths.append(f)
    # An irrelevant report or unsupported CSV should not crash the entire dashboard.
    # Conflicts among VALID runs are still rejected by combine_predictions below.
    bad_sources=[]
    for p in chosen_paths:
        try:
            stat=p.stat()
            frames.append(cached_read(str(p.resolve()),stat.st_mtime_ns,stat.st_size))
        except (ValueError,KeyError,TypeError,UnicodeError,FileNotFoundError,pd.errors.ParserError) as exc:
            bad_sources.append(f'{p}: {exc}')
    for u in uploads or []:
        try:frames.append(cached_uploaded(u.getvalue(),u.name))
        except (ValueError,KeyError,TypeError,UnicodeError,pd.errors.ParserError) as exc:
            bad_sources.append(f'{u.name}: {exc}')
    if bad_sources:
        st.warning('Skipped unsupported prediction source(s):\n\n'+'\n\n'.join('- '+str(x) for x in bad_sources))
    if not frames:
        st.info('Select a compatible `all_predictions.csv` or `all_candidate_day1_predictions.csv` from the sidebar. No valid prediction data is currently loaded.')
        st.stop()
    combined,duplicate_count=combine_predictions(frames)
except Exception as ex:
    st.error(f'Could not load selected predictions: {ex}')
    st.stop()

upcoming=pd.DataFrame()
if future_choice!='None':
    try:
        path=root/future_choice
        stat=path.stat()
        upcoming=cached_upcoming(str(path.resolve()),stat.st_mtime_ns,stat.st_size)
    except (ValueError,KeyError,TypeError,UnicodeError,FileNotFoundError,pd.errors.ParserError) as ex:
        st.warning(f'Upcoming Sunday forecast could not load: {ex}')

names=load_location_names(root)
weekly_source=combined.loc[combined.model.astype(str).str.startswith(WEEKLY_PREFIX)].copy()
model_names=sorted(m for m in combined.model.unique().tolist() if not str(m).startswith(WEEKLY_PREFIX))
if not model_names:
    st.info('Select at least one DAILY model forecast CSV alongside the Sunday-weekly CSV.');st.stop()
preferred='lgbm_stable_recent' if 'lgbm_stable_recent' in model_names else model_names[0]

with st.sidebar:
    st.markdown('### Forecast settings')
    primary=st.selectbox('Primary model',model_names,index=model_names.index(preferred))
    choices=['None']+[m for m in model_names if m!=primary]
    compare_default='weather_median' if 'weather_median' in choices else 'None'
    comparator=st.selectbox('Compare with',choices,index=choices.index(compare_default))
    comparator=None if comparator=='None' else comparator
    weekly_models=sorted(weekly_source.model.unique().tolist())
    if weekly_models:
        weekly_default=WEEKLY_PREFIX+primary
        weekly_selected=st.selectbox('Sunday weekly model',weekly_models,
             index=weekly_models.index(weekly_default) if weekly_default in weekly_models else 0,
             format_func=lambda x:x.replace(WEEKLY_PREFIX,'')+' (Sunday-issued)')
        show_weekly_overlay=st.checkbox('Overlay Sunday forecast on day/week charts',value=False,
             help='Purple dashed line: fixed Sunday 6 PM forecast for Monday through next Sunday. ' 
                  'Daily model updates every evening; compare by target hour, not issuance timestamp.')
    else:
        weekly_selected=None
        show_weekly_overlay=False
    lead_options=sorted(combined.lead_day.unique().tolist())
    horizon=st.selectbox('Forecast lead day',lead_options,index=lead_options.index(1) if 1 in lead_options else 0,
                         format_func=lambda d:f'Day {d} {"— tomorrow" if d==1 else "— later"}')
    quality=st.selectbox('Quality sensitivity',[
        'All (official)','Recorded hours only','Store-days with no inferred zeros',
        'Exclude store-days with 4+ inferred zeros'],index=0,
        help='Filtered cohorts are diagnostic only. They do not prove questionable labels are bad.')
    buckets=st.checkbox('Show current traffic thresholds',value=False)

subset=combined.loc[combined.lead_day.eq(horizon)&combined.model.isin(
    [primary]+([comparator] if comparator else []))].copy()
try:subset=quality_cohort(subset,quality)
except ValueError as ex:
    st.warning(f'{ex}. Reverting to the official unfiltered evaluation.')
    quality='All (official)'
    subset=combined.loc[combined.lead_day.eq(horizon)&combined.model.isin([primary]+([comparator] if comparator else []))].copy()

base=subset.loc[subset.model.eq(primary)].copy()
if base.empty:st.warning('No predictions for this model and lead day.');st.stop()
all_dates=sorted(base.local_date.dropna().unique().tolist())
loc_ids=sorted(base.location_id.astype(str).unique())
site_labels={id:f'{id} · {names.get(id,id)}' if id in names else id for id in loc_ids}

with st.sidebar:
    st.markdown('### Exploration')
    site=st.selectbox('Store',loc_ids,format_func=lambda i:site_labels[i],
                      index=loc_ids.index('AL003') if 'AL003' in loc_ids else 0)
    chosen_range=st.date_input('Target-date range',
                   value=(pd.Timestamp(all_dates[0]).date(),pd.Timestamp(all_dates[-1]).date()),
                   min_value=pd.Timestamp(all_dates[0]).date(),max_value=pd.Timestamp(all_dates[-1]).date())
    if isinstance(chosen_range,(tuple,list)) and len(chosen_range)==2:
        start,end=map(pd.Timestamp,chosen_range)
    else:
        start,end=pd.Timestamp(all_dates[0]),pd.Timestamp(all_dates[-1])

subset=subset.loc[subset.local_date.between(start,end)].copy()
base=subset.loc[subset.model.eq(primary)].copy()
if base.empty:st.warning('No forecasts inside the selected date range.');st.stop()
site_data=subset.loc[subset.location_id.eq(site)].copy()
site_base=site_data.loc[site_data.model.eq(primary)].copy()

# Weekly forecast vintages are NOT filtered by the selected daily lead day.
# Keep them separate from all regular daily scorecards.
weekly_filtered=weekly_source.loc[weekly_source.local_date.between(start,end)].copy()
if weekly_selected:
    weekly_filtered=weekly_filtered.loc[weekly_filtered.model.eq(weekly_selected)].copy()
    try:
        weekly_filtered=quality_cohort(weekly_filtered,quality)
    except ValueError as exc:
        st.warning(f'Weekly quality filter unavailable ({exc}); using all weekly rows.')
weekly_site=weekly_filtered.loc[weekly_filtered.location_id.eq(site)].copy()

src_info=', '.join(sorted(combined.source_run.unique().astype(str)))
coverage=(base[['issued_at']].drop_duplicates().shape[0],base.local_date.nunique(),
          base.location_id.nunique())
st.caption(f'Loaded: {src_info} · {coverage[0]} forecast issuances · {coverage[1]} target dates · '
           f'{coverage[2]} sites · Day {horizon} · {quality}')
if duplicate_count:
    st.warning(f'{duplicate_count:,} identical overlapping forecast rows were deduplicated across selected datasets. Conflicting predictions are never silently merged.')
if coverage[1]<(end-start).days*.65:
    st.warning('Sparse date coverage: this experiment sampled only some forecast issuances. Lines and monthly scores describe sampled dates, not every date in the interval.')
if quality!='All (official)':
    st.info('**Sensitivity view:** one or more evaluation labels have been filtered. Official accuracy includes all observed and reconstructed hours.')

score=metrics(base)
site_score=metrics(site_base)
row=st.columns(6)
row[0].metric('Network Day '+str(horizon)+' MAE',clean_num(score['mae']))
row[1].metric('Typical hourly miss',clean_num(score['median_ae']),'median abs. error')
row[2].metric('Network WAPE',clean_num(score['wape'],'%'))
row[3].metric(f'{primary} · bucket agreement',clean_num(score['bucket_agreement'],'%'))
row[4].metric(f'{primary} · Forecast Labor Alignment',clean_num(score['forecast_labor_alignment_percent'],'%'),
              help='100 × (1 − total absolute difference in forecast-derived vs actual-wash recommended employee-hours / actual-wash recommended employee-hours).')
row[5].metric(f'{site} · {primary} alignment',clean_num(site_score['forecast_labor_alignment_percent'],'%'))

st.caption('Forecast Labor Alignment is a hypothetical company-formula score: forecast-implied employees versus actual-wash recommended employees. Full-week scores sum ALL hourly absolute employee differences, then divide by ALL hourly actual recommendations—not by averaging daily percentages. This is NOT actual staffing performance or a training target. Each score identifies its selected model.')

tab_day,tab_week,tab_sunday,tab_period,tab_sites,tab_compare,tab_quality=st.tabs([
    'Hourly day','Seven-day view','Sunday weekly forecast','Across months',
    'Store reliability','Models & buckets','Data quality & coverage'])

with tab_day:
    st.subheader(f'Day view · {site_labels[site]}')
    dates=sorted(site_base.local_date.unique().tolist())
    last=dates[-1]
    chosen_day=st.selectbox('Target day',options=dates,index=len(dates)-1,
                             format_func=lambda x:pd.Timestamp(x).strftime('%a %b %d, %Y'))
    day=site_data.loc[site_data.local_date.eq(chosen_day)].copy()
    main=day.loc[day.model.eq(primary)]
    if main.empty:st.info('No prediction for this day')
    else:
        m=metrics(main)
        c=st.columns(6)
        for col,(label,val) in zip(c,[('Hour MAE',clean_num(m['mae'])),
                                    ('Median miss',clean_num(m['median_ae'])),
                                    ('Signed bias',clean_num(m['bias'])),
                                    ('Bucket correct',clean_num(m['bucket_agreement'],'%')),
                                    (f'{primary} · alignment',clean_num(m['forecast_labor_alignment_percent'],'%')),
                                    ('Inferred-zero hours',str(int(main.inferred_zero.fillna(False).sum()))) ]):
            col.metric(label,val)
        if show_weekly_overlay and weekly_selected:
            day_monday=pd.Timestamp(chosen_day)-pd.Timedelta(days=pd.Timestamp(chosen_day).weekday())
            weekly_day=sunday_weekly_rows(weekly_site,model=weekly_selected,location=site,monday=day_monday)
            weekly_day=weekly_day.loc[weekly_day.local_date.eq(chosen_day)]
            if not weekly_day.empty:
                day=pd.concat([day,weekly_day],ignore_index=True)
        st.plotly_chart(plot_hours(day,primary,comparator,buckets,
                       weekly_model=weekly_selected if show_weekly_overlay else None,lead_day=horizon),use_container_width=True)
        show_matched_alignment(day,[primary,comparator,weekly_selected if show_weekly_overlay else None],
                               title='Day alignment — compared models',lead_day=horizon)
        st.plotly_chart(hour_error_chart(main),use_container_width=True)
        d=daily_totals(main)
        if not d.empty:
            r=d.iloc[0]
            if bool(r.complete):
                st.caption(f'Full-day total (secondary diagnostic): actual {r.actual:.0f} · predicted {r.predicted:.1f} · error {r.signed_error:+.1f} washes.')
            else:st.warning(f'Daily total NOT scored: only {r.hours}/{r.expected_hours} scheduled hours present.')
        view=main[['hour','carwashes','prediction','signed_error','abs_error',
                   'actual_bucket','predicted_bucket','actual_recommended_employees',
                   'predicted_recommended_employees','employee_difference','inferred_zero']].sort_values('hour').copy()
        view['hour']=view.hour.map(fmt_hour)
        view['actual_bucket']=view.actual_bucket.map(dict(enumerate(RANGES)))
        view['predicted_bucket']=view.predicted_bucket.map(dict(enumerate(RANGES)))
        st.dataframe(view.rename(columns={'carwashes':'actual','signed_error':'predicted_minus_actual'}),
                     hide_index=True,use_container_width=True)

with tab_week:
    st.subheader(f'Week view · {site_labels[site]}')
    weeks=sorted(week_start(site_base.local_date).unique().tolist())
    w=st.selectbox('Week starting Monday',weeks,index=len(weeks)-1,
                   format_func=lambda x: f'{pd.Timestamp(x).strftime("%b %d, %Y")} – {(pd.Timestamp(x)+pd.Timedelta(days=6)).strftime("%b %d")}')
    weekly=site_data.loc[week_start(site_data.local_date).eq(w)].copy()
    wm=metrics(weekly.loc[weekly.model.eq(primary)])
    st.caption(f'{primary} (daily-updated Day {horizon}): {wm["n"]} covered operating hours · '
               f'MAE {clean_num(wm["mae"])} · WAPE {clean_num(wm["wape"],"%")} · '
               f'bucket agreement {clean_num(wm["bucket_agreement"],"%")} · '
               f'Forecast Labor Alignment {clean_num(wm["forecast_labor_alignment_percent"],"%")}')
    expected_week_hours=76
    if wm['n']<expected_week_hours:
        st.warning(f'Only {wm["n"]}/{expected_week_hours} operating hours are available for '
                   'this store-week; any score is for covered hours only, not a full week.')
    if show_weekly_overlay and weekly_selected:
        weekly_line=sunday_weekly_rows(weekly_site,model=weekly_selected,location=site,monday=pd.Timestamp(w))
        if not weekly_line.empty:
            weekly=pd.concat([weekly,weekly_line],ignore_index=True)
    st.plotly_chart(weekly_small_multiples(weekly,primary,pd.Timestamp(w),comparator,buckets,
                         weekly_model=weekly_selected if show_weekly_overlay else None,lead_day=horizon),use_container_width=True)
    show_matched_alignment(weekly,[primary,comparator,weekly_selected if show_weekly_overlay else None],
                           title='Full selected week — alignment by model',lead_day=horizon,
                           period='day')
    st.plotly_chart(weekly_signed_heatmap(weekly,primary,pd.Timestamp(w)),use_container_width=True)
    days=daily_totals(weekly)
    if not days.empty:
        ds=days.loc[days.model.eq(primary)]
        st.caption(f'Complete store-days: {int(ds.complete.sum())} of {len(ds)}. Incomplete days do not receive daily-total scores.')
        st.plotly_chart(daily_traffic_chart(days,primary,comparator),use_container_width=True)

with tab_sunday:
    show_upcoming=False
    if not upcoming.empty:
        show_upcoming=st.toggle('View upcoming Sunday forecast (no actual counts yet)',value=False)
    if show_upcoming:
        st.subheader('Upcoming Sunday-issued forecast · Monday through next Sunday')
        st.info('These are predictions only. Actual washes are not yet available here, so '
                'MAE, bucket agreement and labor alignment are intentionally NOT calculated.')
        future_models=sorted(upcoming.model.unique().tolist())
        future_model=st.selectbox('Weekly forecast model',future_models)
        future_site=st.selectbox('Store for upcoming forecast',sorted(upcoming.location_id.unique()),
                                 format_func=lambda x:site_labels.get(x,x))
        future_issues=sorted(upcoming.issued_at.unique())
        future_issue=st.selectbox('Sunday issuance',future_issues,index=len(future_issues)-1,
             format_func=lambda x:pd.Timestamp(x).tz_convert('America/Chicago').strftime('%b %d, %Y · %I:%M %p'))
        sub=upcoming.loc[(upcoming.model.eq(future_model)) &
                         (upcoming.location_id.eq(future_site)) &
                         (upcoming.issued_at.eq(pd.Timestamp(future_issue)))].copy()
        week_monday=pd.Timestamp(future_issue).tz_convert('America/Chicago').tz_localize(None).normalize()+pd.Timedelta(days=1)
        if len(sub)!=76 or sub.local_date.nunique()!=7:
            st.warning(f'Only {len(sub)}/76 expected hours are present. Check closures or weather forecast coverage.')
        st.plotly_chart(upcoming_seven_days_chart(sub,week_monday,buckets=buckets),use_container_width=True)
        st.markdown('**Forecasted washes per operating hour**')
        table=sub[['local_date','hour','lead_day','prediction']].sort_values(['local_date','hour']).copy()
        table['recommended_employees_if_used']=employee_counts(table.prediction)
        st.dataframe(table,hide_index=True,use_container_width=True)
        st.caption('Employee counts are shown ONLY as a current-policy illustration. '
            'The model predicts wash counts, not labor assignments.')
        st.download_button('Download upcoming seven-day forecast',sub.to_csv(index=False),
                           file_name=f'washcast_upcoming_{future_site}.csv',mime='text/csv')
    else:
        st.subheader('Sunday-issued weekly forecast · Monday through the following Sunday')
        st.caption('One forecast snapshot is issued each Sunday at 6 PM America/Chicago. '
            'It covers seven full operating days. The later days may be less accurate because '
            'the weather forecast is older. These forecasts are not refreshed during the week.')
        if weekly_selected is None or weekly_site.empty:
            st.info('To use this tab, run the Sunday seven-day historical benchmark and load '
                    '`sunday_weekly_*/weekly_predictions.csv` in Prediction sources. '
                    'The regular daily forecasts remain separate.')
        else:
            issues=sorted(weekly_site.issued_at.unique())
            pick=st.selectbox('Issued on Sunday at 6 PM Central',issues,index=len(issues)-1,
                 format_func=lambda x:pd.Timestamp(x).tz_convert('America/Chicago').strftime('%a %b %d, %Y · %I:%M %p'))
            issue=pd.Timestamp(pick)
            start_monday=issue.tz_convert('America/Chicago').tz_localize(None).normalize()+pd.Timedelta(days=1)
            week_sunday=start_monday+pd.Timedelta(days=6)
            chunk=weekly_site.loc[weekly_site.issued_at.eq(issue) &
                  weekly_site.local_date.between(start_monday,week_sunday)].copy()
            # Daily Day-1 forecasts are separate issuances, issued one evening before each target day.
            day1=combined.loc[(combined.model.eq(primary)) & (combined.lead_day.eq(1)) &
                             combined.location_id.eq(site) &
                             combined.local_date.between(start_monday,week_sunday)].copy()
            secondary_day1=combined.loc[(combined.model.eq(comparator)) & (combined.lead_day.eq(1)) &
                             combined.location_id.eq(site) &
                             combined.local_date.between(start_monday,week_sunday)].copy() if comparator else pd.DataFrame()
            try:
                day1=quality_cohort(day1,quality)
                if not secondary_day1.empty:
                    secondary_day1=quality_cohort(secondary_day1,quality)
            except ValueError:
                pass
            show_daily=st.checkbox('Compare against daily refreshed Day 1 model',value=True,
                 key='sunday_weekly_daily_compare')
            observed_days=chunk.local_date.nunique()
            if observed_days<7 or chunk.hour.size != 76:
                st.warning(f'Incomplete labelled week: {observed_days}/7 target days and '
                    f'{len(chunk)}/76 regular operating hours for this store. '
                    'Do not treat a partial observed week as a full-week accuracy score.')
            else:
                st.success('Complete Monday–Sunday observed operating-hour coverage for this store.')
            m=metrics(chunk)
            st.caption(f'Sunday-issued model: **{weekly_selected}** · '
                       f'{len(chunk)}/76 hours · 6 PM Sunday issue · '
                       'Monday–Sunday (partial-week metrics if incomplete).')
            cols=st.columns(5)
            cols[0].metric('Weekly-issued hourly MAE',clean_num(m['mae']))
            cols[1].metric('Median hourly miss',clean_num(m['median_ae']))
            cols[2].metric('WAPE',clean_num(m['wape'],'%'))
            cols[3].metric('Bucket agreement',clean_num(m['bucket_agreement'],'%'))
            cols[4].metric(f'{weekly_selected} · alignment',clean_num(m['forecast_labor_alignment_percent'],'%'))
            display=pd.concat([chunk,day1,secondary_day1],ignore_index=True) if show_daily else chunk
            st.plotly_chart(weekly_small_multiples(display,weekly_selected,start_monday,
                            comparison=primary if show_daily else None,buckets=buckets,
                            extra_comparison=comparator if show_daily else None),use_container_width=True)
            show_matched_alignment(display,[weekly_selected,primary if show_daily else None,
                                             comparator if show_daily else None],
                  title='Sunday-issued versus daily-updated alignment (matched hours)',
                  lead_day=1,period='day')
            ld=[]
            for lead,g in chunk.groupby('lead_day',sort=True):
                ld.append({'day':int(lead),'date':g.local_date.iloc[0].date(),**metrics(g)})
            if ld:
                st.markdown('**Seven-day forecast accuracy by lead day**')
                st.dataframe(pd.DataFrame(ld)[['day','date','n','mae','median_ae','wape',
                     'bucket_agreement','forecast_labor_alignment_percent','bias']],
                     hide_index=True,use_container_width=True)
            if show_daily and not day1.empty:
                try:
                    paired=matched_daily_weekly(day1,chunk)
                    if not paired.empty:
                        daily_abs=np.abs(paired.daily_prediction-paired.carwashes_daily)
                        weekly_abs=np.abs(paired.weekly_prediction-paired.carwashes_weekly)
                        c=st.columns(3)
                        c[0].metric('Matched hours',f'{len(paired):,}')
                        c[1].metric('Weekly-issued MAE',clean_num(weekly_abs.mean()))
                        c[2].metric('Daily Day-1 MAE',clean_num(daily_abs.mean()))
                        p=[]
                        for lead,g in paired.groupby('lead_day',sort=True):
                            y=g.carwashes_weekly.to_numpy()
                            we=employee_counts(g.weekly_prediction)
                            de=employee_counts(g.daily_prediction)
                            real=employee_counts(y)
                            denom=real.sum()
                            p.append({'weekly_lead_day':int(lead),'matched_hours':len(g),
                                      'weekly_mae':float(np.abs(g.weekly_prediction-y).mean()),
                                      'daily_mae':float(np.abs(g.daily_prediction-y).mean()),
                                      'weekly_alignment_percent':100*(1-np.abs(we-real).sum()/denom),
                                      'daily_alignment_percent':100*(1-np.abs(de-real).sum()/denom)})
                        st.markdown(f'**Paired Sunday forecast ({weekly_selected}) versus daily Day 1 ({primary})**')
                        paired_table=pd.DataFrame(p).rename(columns={
                          'weekly_alignment_percent':f'{weekly_selected} alignment (%)',
                          'daily_alignment_percent':f'{primary} daily Day 1 alignment (%)',
                          'weekly_mae':f'{weekly_selected} hourly MAE',
                          'daily_mae':f'{primary} hourly MAE'})
                        st.dataframe(paired_table,hide_index=True,use_container_width=True)
                        st.caption('Compared on the same target store-hours, but from DIFFERENT '
                            'issue times. Daily has newer observations and usually fresher weather; '
                            'the difference measures the value of daily updating, not pure model quality.')
                    else:
                        st.info('No matching daily Day-1 forecasts for this store/week.')
                except ValueError as exc:
                    st.error(f'Unable to make a paired weekly/daily comparison: {exc}')
            st.download_button('Download this Sunday forecast',chunk.to_csv(index=False),
                 file_name=f'washcast_sunday_{issue.tz_convert("America/Chicago").strftime("%Y%m%d")}_{site}.csv',
                 mime='text/csv')

with tab_period:
    st.subheader('Patterns across the validation months')
    choice=st.radio('Geographic scope',['Selected store','All 15 sites combined'],horizontal=True)
    hist=subset.loc[subset.location_id.eq(site)] if choice=='Selected store' else subset
    st.plotly_chart(monthly_chart(hist,lead_day=horizon),use_container_width=True)
    if show_weekly_overlay and weekly_selected:
        matched_weekly=weekly_filtered.loc[weekly_filtered.location_id.eq(site)].copy() if choice=='Selected store' else weekly_filtered.copy()
        shared_period=pd.concat([hist,matched_weekly],ignore_index=True)
        show_matched_alignment(shared_period,[primary,comparator,weekly_selected],
             title='Monthly model alignment on identical target hours',lead_day=horizon,
             period='month')
    h=hist.loc[hist.model.eq(primary)].copy()
    g=h.groupby('local_date',observed=True).agg(mae=('abs_error','mean'),
                bias=('signed_error','mean'),actual_avg=('carwashes','mean'),
                n=('hour','size')).reset_index().sort_values('local_date')
    if not g.empty:
        g['roll_7_forecast_days']=g.mae.rolling(7,min_periods=3).mean()
        fig=go.Figure()
        fig.add_trace(go.Bar(x=g.local_date,y=g.mae,name='Daily hourly MAE',marker_color=BLUE,opacity=.38))
        fig.add_trace(go.Scatter(x=g.local_date,y=g.roll_7_forecast_days,name='7-forecast-day rolling MAE',
                      mode='lines',line=dict(color=ORANGE,width=2.5)))
        fig.update_yaxes(title='Hourly MAE (washes)');fig.update_xaxes(title='Target date')
        fig.update_layout(title='Does error persist, or spike for one day?')
        st.plotly_chart(style_fig(fig,340),use_container_width=True)
        st.caption('Rolling window uses seven evaluated dates, which can span more than seven calendar days when validation origins are sparse.')
    dt=daily_totals(hist)
    if choice=='Selected store':
        if dt.complete.any():
            st.plotly_chart(daily_traffic_chart(dt,primary,comparator),use_container_width=True)
            d_m=daily_metrics(dt.loc[dt.model.eq(primary)])
            st.caption(f'Secondary full-day metric: {d_m["days"]} complete days; MAE {clean_num(d_m["mae"])} washes/day; daily-total WAPE {clean_num(d_m["wape"],"%")}')
    month=month_summary(hist)
    st.dataframe(month[['model','month','n','mae','median_ae','wape','bias',
                        'bucket_agreement','forecast_labor_alignment_percent',
                        'under_recommended_employee_hours','over_recommended_employee_hours']].sort_values(['month','model']),
                 hide_index=True,use_container_width=True)
    st.download_button('Download monthly scorecard',month.to_csv(index=False),file_name='washcast_monthly_scores.csv',mime='text/csv')

with tab_sites:
    st.subheader(f'Location-level reliability · {primary} · daily Day {horizon}')
    cards=store_scorecard(subset)
    current=cards.loc[cards.model.eq(primary)].copy()
    current['location']=current.location_id.map(lambda s:names.get(s,s))
    show=['location_id','location','n','mean_actual','mae','median_ae','wape','bias','p90_ae',
          'bucket_agreement','forecast_labor_alignment_percent',
          'under_recommended_employee_hours','over_recommended_employee_hours',
          'bucket_under','bucket_over']
    st.dataframe(current[show].sort_values('mae',ascending=False),hide_index=True,use_container_width=True)
    fig=go.Figure()
    p=current.sort_values('mae',ascending=True)
    fig.add_trace(go.Bar(x=p.mae,y=p.location_id,orientation='h',name='Hourly MAE',marker_color=BLUE))
    fig.update_layout(title='Mean absolute error by location',yaxis=dict(autorange='reversed'))
    st.plotly_chart(style_fig(fig,420),use_container_width=True)
    fig2=go.Figure(go.Scatter(x=p.mean_actual,y=p.wape,mode='markers+text',text=p.location_id,
                              textposition='top center',marker=dict(size=12,color=BLUE)))
    fig2.update_layout(title='Relative error depends on typical store volume')
    fig2.update_xaxes(title='Actual mean washes/hour');fig2.update_yaxes(title='WAPE (%)')
    st.plotly_chart(style_fig(fig2,350),use_container_width=True)

with tab_compare:
    st.subheader('Paired models & current traffic buckets')
    st.caption('Business traffic ranges are a *secondary* rollout measure. We do not train LightGBM to forecast staffing assignments.')
    if comparator:
        try:
            pairs=paired_predictions(subset,primary,comparator)
            if not pairs.empty:
                ea=(pairs.prediction_a-pairs.carwashes_a).abs()
                eb=(pairs.prediction_b-pairs.carwashes_b).abs()
                st.metric(f'Paired MAE gain: {primary} vs {comparator}',clean_num(eb.mean()-ea.mean()),
                          help='Positive means primary model is more accurate; identical target hours only.')
                st.caption(f'{len(pairs):,} matched forecast instances. This avoids comparing models on different stores, dates or target hours.')
                pairs['gain']=eb-ea
                sg=pairs.groupby('location_id',observed=True).gain.mean().sort_values()
                st.bar_chart(sg.rename('Mean MAE advantage (positive favors primary)'))
            else:st.warning('No overlapping forecast instances for the selected models.')
        except ValueError as ex:st.error(str(ex))
    scope=st.radio('Bucket evaluation scope',['All stores','Selected store'],horizontal=True,key='bucket_scope')
    buck_df=base if scope=='All stores' else site_base
    if not buck_df.empty:
        conf=pd.crosstab(buck_df.actual_bucket,buck_df.predicted_bucket).reindex(
             index=range(4),columns=range(4),fill_value=0)
        fig=go.Figure(go.Heatmap(z=conf.to_numpy(),x=list(RANGES),y=list(RANGES),colorscale='Blues',
                               text=conf.to_numpy(),texttemplate='%{text}',
                               hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Hours: %{z}<extra></extra>'))
        fig.update_xaxes(title='Predicted wash-count bucket');fig.update_yaxes(title='Actual wash-count bucket',autorange='reversed')
        fig.update_layout(title='Current traffic-range agreement (rounded forecasts)')
        st.plotly_chart(style_fig(fig,385),use_container_width=True)
        bc=metrics(buck_df)
        k=st.columns(5)
        k[0].metric(f'{primary} · Forecast Labor Alignment',clean_num(bc['forecast_labor_alignment_percent'],'%'))
        k[1].metric('Correct bucket',clean_num(bc['bucket_agreement'],'%'))
        k[2].metric('Underestimated bucket',clean_num(bc['bucket_under'],'%'))
        k[3].metric('Overestimated bucket',clean_num(bc['bucket_over'],'%'))
        k[4].metric('Actual hours ≥61 washes',str(int(buck_df.carwashes.ge(61).sum())))
        st.caption(f'Employee-hours recommended from actual washes: {bc["recommended_employee_hours"]:,} · '
                   f'Absolute forecast-vs-actual difference: {bc["absolute_employee_hour_difference"]:,} · '
                   f'Under-recommended: {bc["under_recommended_employee_hours"]:,} · '
                   f'Over-recommended: {bc["over_recommended_employee_hours"]:,}')
        st.caption('Alignment = 100 × (1 − Σ |forecast-implied employees − actual-wash recommended employees| / Σ actual-wash recommended employees). Forecasts are rounded half-up. This is a hypothetical operational score, not actual staffing efficiency.')
        # Use matched issuance/store/target hours for fair head-to-head FLA comparison.
        if comparator:
            try:
                pair_scope=paired_predictions(subset if scope=='All stores' else site_data,primary,comparator)
                if not pair_scope.empty:
                    actual_staff=employee_counts(pair_scope.carwashes_a)
                    pred_staff=employee_counts(pair_scope.prediction_a)
                    compare_staff=employee_counts(pair_scope.prediction_b)
                    denom=actual_staff.sum()
                    primary_alignment=100*(1-np.abs(pred_staff-actual_staff).sum()/denom)
                    comparison_alignment=100*(1-np.abs(compare_staff-actual_staff).sum()/denom)
                    st.caption(f'Matched hours ({len(pair_scope):,}): **{primary} {primary_alignment:.2f}%** vs '
                               f'**{comparator} {comparison_alignment:.2f}%** '
                               f'({primary_alignment-comparison_alignment:+.2f} percentage points).')
            except ValueError as ex:
                st.warning(f'Could not make a matched alignment comparison: {ex}')
        # Calibration grouped by *forecast* level avoids conditioning on realized extremes.
        q=buck_df.copy()
        q['forecast_band']=pd.cut(q.prediction,bins=[-1,18,35,50,61,np.inf],
                                  labels=['<18','18–35','35–50','50–61','61+'],right=False)
        cal=q.groupby('forecast_band',observed=True).agg(hours=('hour','size'),
                      mean_predicted=('prediction','mean'),mean_actual=('carwashes','mean')).reset_index()
        st.markdown('**High-volume calibration — group by forecast, not actual**')
        st.dataframe(cal,hide_index=True,use_container_width=True)

with tab_quality:
    st.subheader('Source coverage & uncertain labels')
    st.caption('These flags describe provenance; an inferred zero can represent either a real zero or missing reporting. Do not treat it automatically as corruption.')
    issue=base[['issued_at','local_date']].drop_duplicates()
    issue['month']=issue.local_date.dt.to_period('M').astype(str)
    cover=issue.groupby('month',observed=True).agg(issuances=('issued_at','nunique'),
                           evaluated_target_dates=('local_date','nunique')).reset_index()
    st.dataframe(cover,hide_index=True,use_container_width=True)
    reported=base.inferred_zero.eq(True)
    st.metric('Evaluated inferred-zero hours',str(int(reported.sum())))
    if reported.any():
        d=base.groupby(['location_id','local_date'],observed=True).agg(
            hours=('hour','size'),inferred_zeros=('inferred_zero',lambda a:int(a.eq(True).sum())),
            actual=('carwashes','sum'),mean_abs_error=('abs_error','mean')).reset_index()
        suspicious=d.loc[d.inferred_zeros>0].sort_values(['inferred_zeros','mean_abs_error'],ascending=False)
        st.markdown('**Store-days containing inferred-zero hours**')
        st.dataframe(suspicious,hide_index=True,use_container_width=True)
        st.download_button('Download quality flags',suspicious.to_csv(index=False),file_name='washcast_questionable_store_days.csv',mime='text/csv')
    st.markdown('**Important operating-hour note**')
    st.info('The expected full-day hours in this viewer follow the historical regular schedule: Mon–Sat 8 AM–7 PM, Sun 9 AM–7 PM. Verified early closures (such as a July 4 closing at 5 PM) must be added to the *source calendar* before treating the resulting daily totals as ordinary complete operating days. This dashboard does not silently recode them.')

st.divider()
st.caption('WashCast historical validation viewer · Read-only · As-issued comparisons · Do not host sensitive company data on a public Streamlit server')
