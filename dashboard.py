"""Local Streamlit viewer/controller for as-of carwash forecasting v0.2.6.

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
from plotly.subplots import make_subplots
import streamlit as st

import washcast as core
import asof
from rain_explorer import rank_post_rain_periods, one_issued_forecast
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
    rolling_dir=st.text_input('Rolling backtest results',value='rolling_180d')
    compare_dir=st.text_input('Compare against backtest folder (optional)',value='')
    experiment_dir=st.text_input('Weather ablation / MLP experiments',value='experiments_180d')
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


def common_cli(traffic, weather, archive, closures, timezone, open_hour, sunday_open_hour, close_hour):
    result=['--traffic',traffic,'--timezone',timezone,
            '--open-hour',str(open_hour),'--sunday-open-hour',str(sunday_open_hour),'--close-hour',str(close_hour)]
    if weather.strip():result+=['--weather-csv',weather.strip()]
    if archive.strip():result+=['--archive-csv',archive.strip()]
    if closures.strip():result+=['--closures',closures.strip()]
    return result

train_tab,validation_tab,behavior_tab,forecast_tab,control_tab,rolling_tab,experiments_tab=st.tabs([
    'Training timeline','Holdout validation','How models behave','Next-week forecast','Train / refresh / update','Rolling backtest','Weather & MLP experiments'])

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
        open_hour=st.number_input('Monday–Saturday opening hour (24-hour)',min_value=0,max_value=23,value=8)
    with b:
        sunday_open_hour=st.number_input('Sunday opening hour (24-hour)',min_value=0,max_value=23,value=9)
        close_hour=st.number_input('Closing hour (exclusive)',min_value=1,max_value=24,value=19)
    api_required=not bool(archive.strip() and weather.strip())
    confirmed=st.checkbox('I have authorization and a compatible weather-data license for any live API requests',value=False)
    enabled=not api_required or confirmed
    common=common_cli(traffic,weather,archive,closures,tz,open_hour,sunday_open_hour,close_hour)
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


with rolling_tab:
    st.subheader('Rolling-origin backtest · multiple genuinely out-of-sample forecast weeks')
    st.caption('Each historical issuance uses a freshly fitted model trained only on labels available before that issuance. This test runs from local CSV files and does not contact the weather API.')
    rp=Path(rolling_dir)
    overall=table(rp/'overall_metrics.csv')
    issue_metrics=table(rp/'by_issuance.csv')
    lead_metrics=table(rp/'by_lead_day.csv')
    pair_metrics=table(rp/'by_issue_and_lead.csv')
    location_metrics=table(rp/'by_location.csv')
    weekday_metrics=table(rp/'by_weekday.csv')
    roll_pred=table(rp/'backtest_predictions.csv')
    roll_info=meta(rp/'backtest_info.json')
    if overall is None or issue_metrics is None or roll_pred is None:
        st.info('Run `backtest.py` in PowerShell first, then enter its output folder above.')
        st.code(r'.\.venv\Scripts\python.exe backtest.py --examples real_results_1yr/asof_training_examples.csv --window-days 180 --test-origins 8 --models baseline,boosted,mlp --output rolling_180d',language='powershell')
    else:
        st.write('Rolling window (days):',roll_info.get('window_days','?'),
                 ' · Held-out weekly issuances:',roll_info.get('test_origins','?'))
        st.dataframe(overall,use_container_width=True,hide_index=True)
        st.plotly_chart(px.bar(overall,x='model',y='mae',text='mae',title='Overall backtest MAE'),use_container_width=True)
        st.plotly_chart(px.line(issue_metrics,x='issue_date',y='mae',color='model',markers=True,
                                title='Error by historical forecast issuance'),use_container_width=True)
        if lead_metrics is not None:
            st.plotly_chart(px.line(lead_metrics,x='lead_day',y='mae',color='model',markers=True,
                                    title='Error by lead day across all test weeks'),use_container_width=True)
        if pair_metrics is not None and len(pair_metrics):
            model_for_map=st.selectbox('Inspect issuance × lead-day errors',sorted(pair_metrics.model.unique()),key='rolling_heat_model')
            matrix=pair_metrics[pair_metrics.model==model_for_map]
            heat=px.density_heatmap(matrix,x='lead_day',y='issue_date',z='mae',histfunc='avg',
                                   title=f'{model_for_map} · MAE by forecast week and lead day',
                                   labels={'lead_day':'Lead day','issue_date':'Forecast issued','mae':'MAE'})
            st.plotly_chart(heat,use_container_width=True)
        c1,c2=st.columns(2)
        with c1:
            if location_metrics is not None:
                chosen_model=st.selectbox('Model for site-error analysis',sorted(location_metrics.model.unique()),key='rolling_site_model')
                locations=location_metrics[location_metrics.model==chosen_model].sort_values('mae',ascending=False)
                st.plotly_chart(px.bar(locations,x='location_id',y='mae',title=f'{chosen_model}: MAE by location'),use_container_width=True)
        with c2:
            if weekday_metrics is not None:
                st.plotly_chart(px.line(weekday_metrics,x='dow',y='mae',color='model',markers=True,
                                        title='Error by target weekday (0 = Monday)'),use_container_width=True)
        st.markdown('**Inspect an individual forecast**')
        roll_pred['location_id']=roll_pred.location_id.astype(str)
        site=st.selectbox('Location',sorted(roll_pred.location_id.unique()),key='rolling_site')
        issued=st.selectbox('Forecast issuance',sorted(roll_pred.issue_date.unique()),key='rolling_origin')
        filtered=roll_pred[(roll_pred.location_id==site)&(roll_pred.issue_date==issued)].copy()
        day=st.selectbox('Target local date',sorted(filtered.local_date.unique()),key='rolling_day')
        filtered=filtered[filtered.local_date==day]
        plot=go.Figure()
        first=filtered[filtered.model==filtered.model.iloc[0]].sort_values('hour')
        plot.add_trace(go.Scatter(x=first.hour,y=first.carwashes,mode='lines+markers',name='Actual',line={'width':4}))
        for model_name,g in filtered.groupby('model'):
            g=g.sort_values('hour')
            plot.add_trace(go.Scatter(x=g.hour,y=g.prediction,mode='lines+markers',name=model_name))
        plot.update_layout(xaxis_title='Local hour',yaxis_title='Washes',title=f'Forecast issued {issued} for {day} · {site}')
        st.plotly_chart(plot,use_container_width=True)
        if compare_dir.strip():
            cp=Path(compare_dir.strip())
            other=table(cp/'overall_metrics.csv')
            other_pred=table(cp/'backtest_predictions.csv')
            other_info=meta(cp/'backtest_info.json')
            if other is not None and other_pred is not None:
                keys=['issued_at','location_id','timestamp','carwashes']
                k1=roll_pred[keys].drop_duplicates().astype(str).sort_values(keys).reset_index(drop=True)
                k2=other_pred[keys].drop_duplicates().astype(str).sort_values(keys).reset_index(drop=True)
                if not k1.equals(k2):
                    st.warning('These backtests do NOT contain identical evaluation targets. Compare only if you intentionally accept different test samples.')
                else:
                    st.markdown('**Same-test-rows comparison**')
                    merged=overall[['model','mae']].rename(columns={'mae':f"{roll_info.get('window_days','A')}d MAE"}).merge(
                        other[['model','mae']].rename(columns={'mae':f"{other_info.get('window_days','B')}d MAE"}),on='model',how='outer')
                    st.dataframe(merged,use_container_width=True,hide_index=True)
            else:
                st.warning('Comparison folder does not contain a complete rolling backtest.')
        st.warning('Weekly weather archives cannot separate lead time from target weekday. More daily issuance snapshots are required for that analysis. The rolling backtest is a development evaluation, not a final untouched test set.')


with experiments_tab:
    st.subheader('Weather ablation and chronological MLP tuning')
    st.caption('Offline, matched examples: calendar-only versus weather-informed models on exactly the same held-out forecast hours. '
               'Positive weather gain means weather lowers MAE. This is predictive, not a proof of weather causality.')
    ep=Path(experiment_dir)
    eoverall=table(ep/'overall_metrics.csv')
    egain=table(ep/'weather_gain_overall.csv')
    eregime=table(ep/'weather_gain_by_regime.csv')
    eweek=table(ep/'weather_gain_by_issuance.csv')
    emonth=table(ep/'weather_gain_by_month.csv')
    ereconstructed=table(ep/'weather_gain_by_reconstructed_rain.csv')
    emiss=table(ep/'weather_gain_by_unforecast_rain_event.csv')
    elearning=table(ep/'mlp_learning_curves.csv')
    etune=table(ep/'mlp_tuning.csv')
    einfo=meta(ep/'experiment_info.json')
    eall=table(ep/'experiment_predictions.csv')
    if eoverall is None:
        st.info('Run experiments.py offline first and point the sidebar to its output folder.')
        st.code(r'.\.venv\Scripts\python.exe experiments.py --examples real_results_no_AL001/asof_training_examples.csv --window-days 180 --test-origins 8 --models baseline,boosted --output experiments_180d',language='powershell')
    else:
        st.write('Rolling training window:',einfo.get('window_days','?'),'days · Historical forecast issuances:',einfo.get('test_origins','?'))
        st.dataframe(eoverall,use_container_width=True,hide_index=True)
        st.plotly_chart(px.bar(eoverall,x='variant',y='mae',text='mae',title='Forecast MAE · identical evaluated hours'),use_container_width=True)
        if egain is not None and len(egain):
            st.markdown('**Paired weather improvement (calendar MAE minus weather MAE)**')
            st.dataframe(egain,use_container_width=True,hide_index=True)
            st.caption('Positive numbers are improvements. Negative numbers mean weather worsened the predictions.')
        if eregime is not None and len(eregime):
            st.markdown('**Are rainy conditions getting lost in the overall average?**')
            st.plotly_chart(px.bar(eregime,x='weather_regime',y='weather_gain_washes_per_hour',color='algorithm',
                                   barmode='group',hover_data=['observations','calendar_mae','weather_mae'],
                                   title='Added predictive value of weather by forecast-weather regime'),use_container_width=True)
            st.dataframe(eregime,use_container_width=True,hide_index=True)
            st.caption('Based on archived FORECAST precipitation: target rain >= 0.5 mm/h, otherwise previous 72h rain >= 5 mm. '
                       'An unanticipated storm may be labeled dry here, because the archived forecast did not foresee it.')
        if eweek is not None and len(eweek):
            st.markdown('**Weather gains by forecast week**')
            st.plotly_chart(px.bar(eweek,x='issue_date',y='weather_gain_washes_per_hour',color='algorithm',
                                   barmode='group',hover_data=['observations','calendar_mae','weather_mae'],
                                   title='Weather benefit or harm by historical forecast issuance'),use_container_width=True)
            st.dataframe(eweek,use_container_width=True,hide_index=True)
        if ereconstructed is not None and len(ereconstructed):
            st.markdown('**Reconstructed target-hour rainfall (retrospective diagnostic only)**')
            st.plotly_chart(px.bar(ereconstructed,x='reconstructed_rain_regime',y='weather_gain_washes_per_hour',
                                   color='algorithm',barmode='group',hover_data=['observations'],
                                   title='Weather value when the target hour was actually rainy (reconstructed)'),
                            use_container_width=True)
            st.caption('This groups results AFTER prediction using the separately supplied reconstructed historical weather. '
                       'It is never included in model inputs, and is not a verified station observation.')
        if emiss is not None and len(emiss):
            st.markdown('**Rain that archived forecasts missed**')
            st.dataframe(emiss,use_container_width=True,hide_index=True)
        if emonth is not None and len(emonth):
            st.markdown('**Weather gains by target month**')
            st.plotly_chart(px.bar(emonth,x='target_month',y='weather_gain_washes_per_hour',color='algorithm',
                                   barmode='group',title='Weather contributions by month of target hour'),use_container_width=True)
            st.caption('March versus October needs held-out forecast dates in BOTH months. '
                       'A last-eight-weeks experiment cannot tell us about March.')
        if etune is not None and len(etune):
            st.markdown('**MLP inner-validation tuning selections**')
            st.dataframe(etune,use_container_width=True,hide_index=True)
        if elearning is not None and len(elearning):
            st.markdown('**MLP training vs validation across epochs**')
            issue=st.selectbox('Historical forecast issuance', sorted(elearning.issued_at.unique()),key='exp_origin')
            feature=st.selectbox('Feature version', sorted(elearning.feature_set.unique()),key='exp_feature')
            part=elearning[(elearning.issued_at==issue)&(elearning.feature_set==feature)]
            config=st.selectbox('MLP configuration',sorted(part.candidate.unique()),key='exp_candidate')
            part=part[part.candidate==config]
            long=part.melt(id_vars=['epoch'],value_vars=['inner_training_mae','inner_validation_mae'],
                           var_name='series',value_name='MAE')
            st.plotly_chart(px.line(long,x='epoch',y='MAE',color='series',
                                    title='Chronological inner-validation learning curves'),use_container_width=True)
            st.caption('The best epoch/configuration was chosen only from inner validation within each training fold. '
                       'The outer test week was not used to stop the MLP.')
        if eall is not None and len(eall):
            with st.expander('Inspect predicted versus actual for one held-out forecast'):
                location=st.selectbox('Location code',sorted(eall.location_id.astype(str).unique()),key='exp_site')
                issue=st.selectbox('Issuance date',sorted(eall.issue_date.unique()),key='exp_issue')
                filt=eall[(eall.location_id.astype(str)==location)&(eall.issue_date==issue)]
                target=st.selectbox('Target date',sorted(filt.local_date.unique()),key='exp_target')
                filt=filt[filt.local_date==target]
                fig=go.Figure()
                actual=filt[filt.variant==filt.variant.iloc[0]].sort_values('hour')
                fig.add_trace(go.Scatter(x=actual.hour,y=actual.carwashes,name='Actual',mode='lines+markers',line={'width':4}))
                for v,g in filt.groupby('variant'):
                    g=g.sort_values('hour')
                    fig.add_trace(go.Scatter(x=g.hour,y=g.prediction,name=v,mode='lines+markers'))
                fig.update_layout(title=f'{location} · {target} · issued {issue}',xaxis_title='Local hour',yaxis_title='Washes')
                st.plotly_chart(fig,use_container_width=True)
        with st.expander('Investigate post-rain errors · traffic and weather timeline'):
            st.caption('Ranks the post-rain hours where weather-aware boosting underperforms the historical '
                       'baseline. Uses saved experiment_predictions.csv only; no API requests or retraining.')
            postrain = rank_post_rain_periods(eall)
            if postrain.empty:
                st.info('No matched post-rain baseline and boosted-weather forecasts are available in this experiment.')
            else:
                st.markdown('**Find the worst post-rain location-days**')
                display_cols = ['issue_date', 'location_id', 'local_date', 'hours',
                                'baseline_mae', 'weather_mae', 'extra_mae_vs_baseline', 'weather_signed_bias']
                st.dataframe(postrain[display_cols].head(40).round(2), use_container_width=True, hide_index=True)
                st.caption('Positive extra MAE means boosted weather did worse than baseline. '
                           'Signed bias is prediction minus actual; a negative value indicates underprediction. '
                           'Only hours labeled \'Rain in previous 72h\' contribute to this ranking.')
                chosen = st.selectbox(
                    'Inspect location and target day (sorted by additional post-rain error)',
                    options=list(range(len(postrain))),
                    format_func=lambda j: (
                        f"{postrain.iloc[j]['local_date']} · {postrain.iloc[j]['location_id']} "
                        f"· issued {postrain.iloc[j]['issue_date']} · "
                        f"extra MAE {postrain.iloc[j]['extra_mae_vs_baseline']:+.2f}"),
                    key='postrain_selected')
                selected = postrain.iloc[chosen]
                period = one_issued_forecast(eall, str(selected['issue_date']), str(selected['location_id']))
                if period.empty:
                    st.warning('No detailed predictions found for this location and issuance.')
                else:
                    period = period.copy()
                    period['local_clock'] = (pd.to_datetime(period['timestamp'], utc=True)
                                             .dt.tz_convert('America/Chicago').dt.tz_localize(None))
                    plot = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                         row_heights=[0.57, 0.19, 0.24], vertical_spacing=.075,
                                         subplot_titles=('Actual traffic compared with model forecasts',
                                                         'Rain forecast for each target hour (mm)',
                                                         'Forecast rainfall accumulated over preceding 72h (mm)'))
                    plot.add_trace(go.Scatter(x=period.local_clock, y=period.carwashes,
                                              name='Actual washes', mode='lines+markers',
                                              line={'width': 4, 'color': '#222222'}), row=1, col=1)
                    for name, label, dash in [
                        ('prediction_baseline', 'Historical baseline', 'dash'),
                        ('prediction_boosted_calendar', 'Boosted, calendar only', 'dot'),
                        ('prediction_boosted_weather', 'Boosted, with weather', 'solid'),
                        ('prediction_mlp_weather', 'MLP, with weather', 'longdash'),
                    ]:
                        if name in period.columns:
                            plot.add_trace(go.Scatter(x=period.local_clock, y=period[name],
                                                      name=label, mode='lines+markers',
                                                      line={'dash': dash}), row=1, col=1)
                    if 'forecast_rain_target_mm' in period:
                        plot.add_trace(go.Bar(x=period.local_clock, y=period.forecast_rain_target_mm,
                                              name='Archived forecast rain', opacity=.72), row=2, col=1)
                    if 'forecast_rain_prior_72h_mm' in period:
                        plot.add_trace(go.Scatter(x=period.local_clock, y=period.forecast_rain_prior_72h_mm,
                                                  mode='lines+markers', name='Prior 72h forecast rain'), row=3, col=1)
                    history_path = st.text_input('Optional reconstructed historical weather CSV (for earlier rainfall context)',
                                                 value='', key='postrain_history_csv',
                                                 help='For example: weather_data/daily_partial/historical_weather.csv. '
                                                      'Only read locally; never contacts a weather service.')
                    if history_path.strip():
                        histfile = Path(history_path.strip())
                        if histfile.is_file():
                            try:
                                h = pd.read_csv(histfile,
                                                usecols=lambda c: c in {'timestamp','location_id','precipitation'})
                                if {'timestamp','location_id','precipitation'}.issubset(h.columns):
                                    h = h[h.location_id.astype(str) == str(selected['location_id'])].copy()
                                    h['timestamp'] = pd.to_datetime(h.timestamp, utc=True, errors='coerce')
                                    begin = pd.to_datetime(period.timestamp.min(), utc=True) - pd.Timedelta(hours=72)
                                    finish = pd.to_datetime(period.timestamp.max(), utc=True)
                                    h = h[h.timestamp.between(begin, finish)]
                                    if len(h):
                                        h['local_clock'] = h.timestamp.dt.tz_convert('America/Chicago').dt.tz_localize(None)
                                        plot.add_trace(go.Scatter(x=h.local_clock, y=h.precipitation,
                                                                  name='Reconstructed historical rain',
                                                                  mode='lines', line={'dash': 'dash'}), row=2, col=1)
                                    else:
                                        st.info('No matching reconstructed weather observations in the selected time window.')
                                else:
                                    st.warning('Historical weather file needs timestamp, location_id, precipitation columns.')
                            except Exception as error:
                                st.warning(f'Could not load the optional historical weather CSV: {error}')
                        else:
                            st.warning('Historical weather file not found at that path.')
                    plot.update_yaxes(title_text='Washes/hour', rangemode='tozero', row=1, col=1)
                    plot.update_yaxes(title_text='mm/h', rangemode='tozero', row=2, col=1)
                    plot.update_yaxes(title_text='mm', rangemode='tozero', row=3, col=1)
                    plot.update_xaxes(title_text='Local date/time · America/Chicago', row=3, col=1)
                    plot.update_layout(height=790, hovermode='x unified', legend={'orientation':'h', 'y':-0.12},
                                       margin={'t':65,'b':90})
                    st.plotly_chart(plot, use_container_width=True)
                    st.markdown('**Hourly details**')
                    show = ['local_date', 'hour', 'weather_regime', 'carwashes',
                            'prediction_baseline', 'prediction_boosted_calendar', 'prediction_boosted_weather',
                            'forecast_rain_target_mm', 'forecast_rain_prior_72h_mm', 'inferred_zero']
                    st.dataframe(period[[c for c in show if c in period.columns]].round(2),
                                 use_container_width=True, hide_index=True)
                    st.caption('The plotted rainfall used for predictions is the archived forecast known when '
                               'the prediction was issued. Reconstructed historical rain, if supplied, is '
                               'retrospective context, NOT a model input. A 72-hour accumulation alone does '
                               'not show exactly when it last rained. Predictions cover only the held-out '
                               'forecast horizon, not the full three preceding days.')
        st.warning('This is development/validation, not a new untouched test set. '
                   'Weekly archived forecasts still confound lead horizon with weekday. '
                   'Check inferred zeros and closures before interpreting extreme weeks.')
