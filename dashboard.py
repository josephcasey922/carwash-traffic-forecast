"""Local Streamlit viewer/controller for as-of carwash forecasting v0.2.

Run: streamlit run dashboard.py
All files, weather caches, and model artifacts stay on the computer hosting it.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import washcast as core
import asof
import sklearn
if 'lead_day' not in core.NUM:
    core.NUM.append('lead_day')

st.set_page_config(page_title='WashCast · Traffic Forecasting',layout='wide',page_icon='🚗')
st.title('WashCast · Hourly traffic forecasting')
st.caption('Prototype: forecast car washes, compare models, and inspect which historical information was available when predictions were issued.')

with st.sidebar:
    st.header('Project folders')
    model_dir=st.text_input('Training results',value='asof_results')
    forward_dir=st.text_input('Forecast results',value='example_forward')
    st.caption('Use absolute paths if Streamlit starts outside the project folder. Do not upload company CSVs to a public Streamlit deployment.')


def table(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def meta(path: Path):
    return json.loads(path.read_text()) if path.exists() else {}


def chart_traffic(df: pd.DataFrame, date_col: str, ycols: list[str], title: str, labels=None):
    long=df.melt(id_vars=date_col,value_vars=ycols,var_name='Series',value_name='Value')
    fig=px.line(long,x=date_col,y='Value',color='Series',markers=True,title=title)
    fig.update_layout(hovermode='x unified',legend_title_text='')
    return fig


def run_cli(args: list[str]):
    with st.spinner('Running locally; see output below after it completes'):
        process=subprocess.run([sys.executable,'asof.py',*args],capture_output=True,text=True)
    if process.returncode:
        st.error(f'Command failed (exit code {process.returncode})')
    else:
        st.success('Command completed successfully')
    st.code((process.stdout+'\n'+process.stderr)[-25000:],language='text')
    return process.returncode==0


def common_cli(traffic, weather, archive, closures, timezone, open_hour, close_hour):
    result=['--traffic',traffic,'--timezone',timezone,
            '--open-hour',str(open_hour),'--close-hour',str(close_hour)]
    if weather.strip():result+=['--weather-csv',weather.strip()]
    if archive.strip():result+=['--archive-csv',archive.strip()]
    if closures.strip():result+=['--closures',closures.strip()]
    return result

train_tab,validation_tab,behavior_tab,forecast_tab,control_tab=st.tabs([
    'Training timeline','Holdout validation','How models behave','Next-week forecast','Train / refresh / update'])

folder=Path(model_dir)
info=meta(folder/'model_info.json')
comp=table(folder/'model_comparison.csv')
val=table(folder/'validation_predictions.csv')
horizon=table(folder/'horizon_metrics.csv')
forecast_path=Path(forward_dir)
forward=table(forecast_path/'hourly_forecast.csv')
daily=table(forecast_path/'daily_forecast.csv')
forecast_info=meta(forecast_path/'forecast_info.json')

with train_tab:
    st.subheader('Training means predicting *from* a past forecast origin')
    st.markdown('At each historical issuance, the model receives past traffic and weather, plus the **archived forecast known at that time**. It tries to predict the next 1–7 days of hourly traffic. The true historical car-wash count is only the training label.')
    st.graphviz_chart('''digraph G { rankdir=LR; node [shape=box,style="rounded,filled",fillcolor="#EDF2F7",color="#718096"];
      A [label="At forecast issue\nPast washes + past weather"];
      B [label="Archived issued forecast\nFuture weather for 1–7 days"];
      C [label="Location + calendar\nhour / weekday / lead day"];
      D [label="Model"];
      E [label="Predicted hourly washes"];
      F [label="Observed wash counts\nlabels from later"];
      A->D; B->D; C->D; D->E; E->F [label="compare / learn"];
    }''',use_container_width=True)
    if not info:
        st.info('Train a model below or point the sidebar to a saved model folder.')
    else:
        a,b,c,d=st.columns(4)
        a.metric('Trained sites',info.get('site_count','–'))
        b.metric('Training examples',f"{info.get('training_examples',0):,}")
        c.metric('Held-out forecasts',f"{info.get('validation_examples',0):,}")
        d.metric('Best model',info.get('best_model','–'))
        st.write('Last full refit (UTC):', info.get('last_full_fit_at_utc') or 'Not recorded in older artifacts')
        st.write('Last incremental MLP update (UTC):', info.get('last_incremental_update_at_utc') or 'Never / not recorded')
        st.write('Latest training target (UTC):', info.get('last_training_target_utc','Unknown'))
        st.write('Number of incremental updates:', info.get('incremental_updates',0))
        st.write('Selected issuance dates:',info.get('origin_dates'))
        st.write('Held-out issuance dates:',', '.join(info.get('validation_origins',[])))
        st.caption('Training rows with target times reaching validation issuance are excluded to prevent target leakage. One real target can appear under multiple earlier forecast origins, with different forecast weather.')
        if val is not None and len(val):
            val['issued_at']=pd.to_datetime(val.issued_at,utc=True)
            val['timestamp']=pd.to_datetime(val.timestamp,utc=True)
            issue=st.selectbox('Illustrative historical forecast origin',sorted(val.issued_at.unique()),format_func=lambda t:str(pd.Timestamp(t)))
            subset=val[val.issued_at==issue]
            tz=info.get('timezone','America/Chicago')
            unique_days=sorted(subset.timestamp.dt.tz_convert(tz).dt.date.unique())
            start=pd.Timestamp(issue)
            end=subset.timestamp.max()
            fig=go.Figure()
            fig.add_vrect(x0=start-pd.Timedelta(days=3),x1=start,fillcolor='lightblue',opacity=.25,line_width=0,
                          annotation_text='Known history',annotation_position='top left')
            fig.add_vrect(x0=start,x1=end,fillcolor='gold',opacity=.14,line_width=0,
                          annotation_text='Future prediction targets',annotation_position='top left')
            fig.add_shape(type='line',x0=start,x1=start,y0=-0.2,y1=0.2,line={'dash':'dash','color':'#475569'})
            fig.add_annotation(x=start,y=0.15,text='Forecast issuance',showarrow=False,yshift=12)
            fig.add_trace(go.Scatter(x=[start-pd.Timedelta(days=3),end],y=[0,0],mode='lines',line={'color':'#718096'},showlegend=False))
            fig.update_layout(title='What was known versus unknown at issuance',yaxis={'visible':False},height=230)
            st.plotly_chart(fig,use_container_width=True)

with validation_tab:
    st.subheader('Chronological test on unseen forecast issuance dates')
    if comp is None or val is None:
        st.info('Run `asof.py fit` to populate model comparisons and holdout predictions.')
    else:
        st.dataframe(comp,use_container_width=True,hide_index=True)
        st.plotly_chart(px.bar(comp,x='model',y='mae',title='Validation MAE (washes per hour)',text='mae'),use_container_width=True)
        if horizon is not None:
            st.plotly_chart(px.line(horizon,x='lead_day',y='mae',color='model',markers=True,
                                    title='Error by forecast lead day'),use_container_width=True)
        chosen_site=st.selectbox('Site',sorted(val.location_id.astype(str).unique()),key='val_site')
        selected_origin=st.selectbox('Forecast issued on',sorted(val.issued_at.unique()),key='val_origin')
        model_choices=[c[5:] for c in val.columns if c.startswith('pred_')]
        models_to_view=st.multiselect('Models to show',model_choices,default=model_choices[:3])
        v=val[(val.location_id.astype(str)==chosen_site)&(val.issued_at==selected_origin)].copy()
        v['timestamp']=pd.to_datetime(v.timestamp,utc=True)
        day=st.selectbox('Target day',sorted(v.local_date.unique()),key='val_day')
        view=v[v.local_date==day].sort_values('timestamp')
        g=go.Figure()
        g.add_trace(go.Scatter(x=view.hour,y=view.carwashes,mode='lines+markers',name='Actual washes',line={'width':4}))
        for m in models_to_view:
            g.add_trace(go.Scatter(x=view.hour,y=view[f'pred_{m}'],mode='lines+markers',name=m))
        g.update_layout(title=f'Unseen hourly forecasts at site {chosen_site} on {day}',
                        xaxis_title='Local hour (24-hour)',yaxis_title='Washes',hovermode='x unified')
        st.plotly_chart(g,use_container_width=True)
        st.caption('Validation rows were held out by issuance date. The training cutoff also excludes any training label from the validation period. Historical weather preceding issuance is supplied or reconstructed; forecast weather comes from as-of snapshots.')

with behavior_tab:
    st.subheader('What each model learns')
    explanations={
      'baseline':'Median of past washes for the same location, weekday and hour; fallback to location-hour, location, or overall median. No weather effect.',
      'poisson':'Regularized Poisson regression: log(expected washes) is a weighted sum of encoded features. Coefficients express proportional associations.',
      'negbin':'Negative-binomial regression: like Poisson but assumes additional count variance. Dispersion is estimated from a preliminary Poisson fit.',
      'boosted':'Gradient-boosted decision trees with Poisson count loss. Each tree corrects residual structure from earlier trees, allowing nonlinear weather effects.',
      'mlp':'Two hidden layers (24 and 12 units) with ReLU activations. Trained using Adam and squared-error loss on hourly counts.'}
    if not info:
        st.info('Fit models to see learned behavior.')
    else:
        selected=st.selectbox('Model to inspect',info['models'])
        st.write(explanations.get(selected,''))
        model_file=folder/'models'/f'{selected}.joblib'
        if model_file.exists():
            st.caption('Only load .joblib models created locally by this project, not untrusted files.')
            try:
                m=asof.load_saved_model(folder,selected,info)
            except (RuntimeError, FileNotFoundError) as exc:
                st.error(str(exc))
                m=None
            if m is not None:
                if selected=='mlp':
                    losses=getattr(m.named_steps['model'],'loss_curve_',[])
                    if len(losses):
                        st.plotly_chart(px.line(x=np.arange(1,len(losses)+1),y=losses,
                                labels={'x':'Training iteration','y':'Training loss'},title='MLP optimization during final refit'),use_container_width=True)
                elif selected in ('poisson','negbin'):
                    if selected=='poisson':
                        names=m.named_steps['features'].get_feature_names_out()
                        coeffs=m.named_steps['model'].coef_
                    else:
                        names=m.trans.get_feature_names_out()
                        coeffs=m.params[1:]
                    d=pd.DataFrame({'feature':names,'coefficient':coeffs})
                    d=d.reindex(d.coefficient.abs().sort_values(ascending=False).index).head(15)
                    st.plotly_chart(px.bar(d,x='coefficient',y='feature',orientation='h',title='Largest standardized / encoded feature coefficients'),use_container_width=True)
                    st.caption('One-hot and standardized values; association is not causation. Correlated weather variables can make individual coefficients unstable.')
                elif selected=='baseline':
                    st.write('Learned location / weekday / hour median combinations:',len(m.exact))
                else:
                    st.write('This model contains 130 sequential boosting iterations, each a collection of small regression trees (subject to early stopping).')
                sample_file=folder/'asof_training_examples.csv'
                if sample_file.exists():
                    samples=pd.read_csv(sample_file)
                    if len(samples):
                        st.markdown('**Model response to changing hourly precipitation**')
                        st.caption('This is a what-if model behavior curve, not a causal effect or a historical experiment. Other feature values are held fixed.')
                        idx=st.number_input('Representative training row',min_value=0,max_value=len(samples)-1,value=0,step=1)
                        sample=samples.iloc[[idx]].copy()
                        rainfall=np.linspace(0,8,25)
                        examples=pd.concat([sample]*len(rainfall),ignore_index=True)
                        examples['precipitation']=rainfall
                        preds=core.predict_model(selected,m,examples)
                        st.plotly_chart(px.line(x=rainfall,y=preds,labels={'x':'Hypothetical precipitation (mm/hour)','y':'Expected washes'},
                                       title=f'{selected}: modeled prediction versus rain'),use_container_width=True)

with forecast_tab:
    st.subheader('Next 7 days: expected washes')
    if forward is None or daily is None:
        st.info('Run the `forecast` command below to create a forward forecast without retraining.')
    else:
        st.warning('No calibrated prediction intervals yet. Plots show point forecasts, not validated low/high ranges.',icon='⚠️')
        available=[c[5:] for c in forward.columns if c.startswith('pred_')]
        model=st.selectbox('Saved model',available,index=available.index(forecast_info.get('model')) if forecast_info.get('model') in available else 0,key='forward_model')
        sites=['All locations']+sorted(forward.location_id.astype(str).unique())
        site=st.selectbox('Location',sites,key='forward_site')
        metric=st.radio('Forecast metric',['Washes','Illustrative revenue proxy'],horizontal=True)
        dollars=float(forecast_info.get('revenue_per_wash_assumption',0))
        if metric=='Illustrative revenue proxy' and dollars<=0:
            st.info('Supply a positive `--revenue-per-wash` assumption to display an illustrative revenue proxy. Wash counts alone cannot determine actual revenue.')
        col=f'pred_{model}'
        d=daily.copy()
        h=forward.copy()
        if site!='All locations':
            d=d[d.location_id.astype(str)==site]
            h=h[h.location_id.astype(str)==site]
        if metric=='Washes':
            d['display_value']=d[col]
            h['display_value']=h[col]
        else:
            d['display_value']=d[col]*dollars
            h['display_value']=h[col]*dollars
        group=d.groupby('local_date',as_index=False).display_value.sum()
        st.metric('Forecast total over displayed dates',f"{'$' if metric!='Washes' else ''}{group.display_value.sum():,.0f}")
        ylabel='Illustrative dollars' if metric!='Washes' else 'Expected car washes'
        fig=px.bar(group,x='local_date',y='display_value',title=f'Daily {ylabel.lower()} — {model}',labels={'display_value':ylabel,'local_date':'Date'})
        st.plotly_chart(fig,use_container_width=True)
        days=sorted(h.local_date.unique())
        target_day=st.selectbox('Inspect hourly profile',days,key='day_forecast')
        v=h[h.local_date==target_day].groupby('hour',as_index=False).display_value.sum()
        st.plotly_chart(px.line(v,x='hour',y='display_value',markers=True,title=f'Hourly pattern: {target_day}',labels={'hour':'Local hour','display_value':ylabel}),use_container_width=True)
        if site!='All locations':
            w=h[h.local_date==target_day].sort_values('hour')
            g=go.Figure()
            g.add_trace(go.Scatter(x=w.hour,y=w.temperature_2m,name='Temperature °C'))
            g.add_trace(go.Bar(x=w.hour,y=w.precipitation,name='Precipitation mm',yaxis='y2',opacity=.5))
            g.update_layout(title='Weather forecast used for this site and date',xaxis_title='Local hour',
                            yaxis_title='Temperature °C',yaxis2={'title':'Rain mm','overlaying':'y','side':'right'},barmode='overlay')
            st.plotly_chart(g,use_container_width=True)
        if metric!='Washes':
            st.caption(f'Revenue proxy = forecasted washes × ${dollars:,.2f} assumed per wash. This is not an actual revenue forecast: memberships, discounts, and transaction mix are not known.')
        st.caption(f"Weather source: {forecast_info.get('weather_source','not recorded')}")
        st.download_button('Download daily forecast CSV',daily.to_csv(index=False),'daily_forecast.csv','text/csv')
        st.download_button('Download hourly forecast CSV',forward.to_csv(index=False),'hourly_forecast.csv','text/csv')

with control_tab:
    st.subheader('Train and run models using local CSV files')
    st.warning('Open-Meteo free hosted weather APIs are non-commercial. For company use, obtain approved/licensed access before making API calls. Offline demo requires no API.')
    traffic=st.text_input('Traffic CSV path',value='demo/traffic.csv')
    weather=st.text_input('Past weather CSV (blank = API)',value='demo/weather.csv')
    archive=st.text_input('Archived forecast CSV (blank = Single Runs API for fit/update; live API for forward)',value='demo/archive_forecasts.csv')
    closures=st.text_input('Closure dates CSV (optional)',value='demo/closures.csv')
    tz=st.text_input('Site timezone',value=info.get('timezone','America/Chicago'))
    a,b=st.columns(2)
    with a:
        open_hour=st.number_input('Open hour (24-hour)',min_value=0,max_value=23,value=7)
    with b:
        close_hour=st.number_input('Close hour (exclusive)',min_value=1,max_value=24,value=19)
    api_required=not bool(archive.strip() and weather.strip())
    confirmed=st.checkbox('I have authorization and a compatible weather-data license for any live API requests',value=False)
    enabled=not api_required or confirmed
    common=common_cli(traffic,weather,archive,closures,tz,open_hour,close_hour)
    st.markdown('#### Full fit / rebuild')
    a,b,c=st.columns(3)
    with a:start_date=st.date_input('First issuance date',value=date(2026,8,20))
    with b:end_date=st.date_input('Last issuance date',value=date(2026,10,27))
    with c:origin_step=st.selectbox('Origin every N days',[1,3,7],index=2)
    validation=st.number_input('Last N origins for holdout',min_value=1,max_value=28,value=2)
    chosen=st.multiselect('Models',core.ALL_MODELS,default=['baseline','mlp'])
    if info:
        saved_sk=info.get('runtime_versions',{}).get('scikit_learn')
        if saved_sk and saved_sk!=sklearn.__version__:
            st.warning(f'Saved models were trained using scikit-learn {saved_sk}; current environment has {sklearn.__version__}. Full refit is required before safe loading.')
        elif not saved_sk:
            st.info('These older saved artifacts do not record their dependency versions. Refit locally if model loading reports a version mismatch.')
    if st.button('Fit models and evaluate',disabled=not enabled or not chosen):
        cmd=['fit',*common,'--start-date',str(start_date),'--end-date',str(end_date),'--origin-step',str(origin_step),
             '--validation-origins',str(validation),'--models',','.join(chosen),'--output',model_dir]
        if run_cli(cmd):st.rerun()
    st.markdown('#### Daily forecast (no retraining)')
    origin_date=st.date_input('Forecast issued on',value=date(2026,11,8))
    dollars=st.number_input('Assumed $/wash — optional, revenue proxy only',min_value=0.0,max_value=10000.0,value=0.0,step=0.5)
    available_models=info.get('models',[])
    default_index=available_models.index('mlp') if 'mlp' in available_models else 0
    target_model=st.selectbox('Model for new forecasts (loads ONLY this model)', available_models, index=default_index) if available_models else None
    if st.button('Produce seven-day forecast using selected saved model',disabled=not enabled or not target_model):
        cmd=['forecast',*common,'--model-dir',model_dir,'--model',target_model,'--origin-date',str(origin_date),
             '--revenue-per-wash',str(dollars),'--output',forward_dir]
        if run_cli(cmd):st.rerun()
    st.markdown('#### Incrementally update MLP (not other models)')
    st.caption('Uses newly available labeled forecast examples that have not been used in prior full fits or updates. The existing preprocessing remains fixed; periodically rebuild from scratch.')
    with st.expander('MLP partial-update settings'):
        u_start=st.date_input('First new issuance date',value=date(2026,10,28))
        u_end=st.date_input('Last new issuance date',value=date(2026,11,3))
        if st.button('Update MLP with newly labeled examples',disabled=not enabled or 'mlp' not in info.get('models',[])):
            cmd=['update',*common,'--model-dir',model_dir,'--start-date',str(u_start),'--end-date',str(u_end),'--origin-step','1']
            if run_cli(cmd):st.rerun()
    st.markdown('**Typical cadence:** refresh next-week forecasts daily without retraining; optionally update the MLP after new actual counts arrive; run a full chronological refit weekly/monthly or after a substantial operational change.')
