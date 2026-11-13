"""Issue a Sunday 6 PM / Monday-through-Sunday LightGBM count forecast.

Train on only prior labelled weekly issues (same feature sets as the tested
weekly benchmark), then derive future weather/storm/recent-demand predictors
strictly as-of the selected Sunday issuance. Future traffic labels are never
used for either training or features. Works offline with archived weather.

The resulting future file deliberately has NO 'carwashes' column: the dashboard
must never score unavailable actual traffic as zeros.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sunday_issue(value: str) -> pd.Timestamp:
    day = date.fromisoformat(value)
    if day.weekday() != 6:
        raise ValueError('--issue-date must be Sunday in America/Chicago')
    import asof
    return asof.issue_time(day, 'America/Chicago', 18).tz_convert('UTC')


def training_cutoff(frame: pd.DataFrame, issue: pd.Timestamp, days: int) -> pd.DataFrame:
    if days <= 0:
        raise ValueError('--window-days must be positive')
    train = frame.loc[(frame.issued_at < issue) & (frame.timestamp < issue) &
                      (frame.issued_at >= issue-pd.Timedelta(days=days))].copy()
    if len(train) < 200 or train.issued_at.nunique() < 8:
        raise ValueError(f'Insufficient prior weekly history: {len(train)} rows, '
                         f'{train.issued_at.nunique()} issuances')
    if not train.lead_day.between(1, 7).all():
        raise ValueError('Training examples must be the seven-day Sunday-issued dataset')
    if (train.issued_at.dt.tz_convert('America/Chicago').dt.dayofweek != 6).any():
        raise ValueError('Training examples contain non-Sunday issuances')
    return train


def future_recent_features(history: pd.DataFrame, future: pd.DataFrame,
                           issue: pd.Timestamp) -> pd.DataFrame:
    """Future dummy labels exist only inside feature derivation; never exported or fit.

    Recent-demand features for this Sunday reference only days before Sunday.
    Source historical labels are pre-cutoff, so no outcome of a forecasted
    future hour can enter this operation.
    """
    import recent_demand_features as rd
    h=history.loc[(history.timestamp < issue) &
                  (history.issued_at < issue) &
                  (history.timestamp >= issue-pd.Timedelta(days=365))].copy()
    placeholder=future.copy()
    placeholder['carwashes']=0.0    # dummy, never used for training or saved
    placeholder['inferred_zero']=False
    merged=pd.concat([h,placeholder],ignore_index=True)
    features,_=rd.derive_features(merged,progress_every=250)
    recent=features.loc[features.issued_at.eq(issue)].copy()
    if len(recent) != len(future):
        raise RuntimeError('Unexpected count of future recent-demand feature rows')
    return recent


def generate(args) -> pd.DataFrame:
    # Defer project dependencies so standalone calendar/cutoff tests can run
    # without importing the user's entire forecasting repository.
    import backtest
    import asof
    import demand_feature_ablation as ablation
    import demand_model_benchmark as dm
    import model_tuning as tuning
    import recent_demand_features as rd
    import storm_features as storm
    import storm_loss_experiment as sl
    import washcast as core
    if args.timezone != 'America/Chicago':
        raise ValueError('Sunday weekly scheduling is currently fixed to America/Chicago')
    issue=sunday_issue(args.issue_date)
    variants=tuple(x.strip() for x in args.models.split(',') if x.strip())
    permitted={'weather_median','lgbm_stable_recent','lgbm_full_recent'}
    if not variants or len(set(variants))!=len(variants) or not set(variants).issubset(permitted):
        raise ValueError('Allowed models: weather_median,lgbm_stable_recent,lgbm_full_recent')
    if args.threads < 1:
        raise ValueError('threads must be positive')
    frame=dm.load_data(args.examples,args.storm_features,args.recent_demand_features)
    train=training_cutoff(frame,issue,args.window_days)
    # IMPORTANT: the source CSV might contain later observations. Future rows
    # are excluded from site availability and from the rolling-demand history.
    traffic,closures,_=asof.active_traffic(args)
    traffic=traffic.loc[traffic.timestamp < issue].copy()
    if traffic.empty:
        raise ValueError('No traffic records available before this Sunday issuance')
    hist_weather=core.load_weather(args.historical_weather_csv)
    archive=asof.load_archive_csv(args.archive_csv)
    archive=archive.loc[archive.issued_at.eq(issue)].copy()
    if archive.empty:
        raise ValueError(f'No archived weather for Sunday {args.issue_date} 6 PM Central')
    day=date.fromisoformat(args.issue_date)
    future=asof.make_rows(traffic,hist_weather,archive,[day],args.timezone,18,7,8,19,
                          closures,False,sunday_open_hour=9)
    if not future.lead_day.between(1,7).all():
        raise RuntimeError('Future target calendar extends beyond next Sunday')
    st=storm.enrich_examples(future,hist_weather,archive)
    recent=future_recent_features(backtest.load_examples(args.examples),future,issue)
    for extra,needed in [(st,list(storm.FEATURES)),(recent,list(rd.FEATURES))]:
        future=future.merge(extra[['issued_at','location_id','timestamp']+needed],
                            on=['issued_at','location_id','timestamp'],
                            how='left',validate='one_to_one')
    overrides=dm._selected_parameters(args.selected_configs)
    all_out=[]
    for model in variants:
        if model=='weather_median':
            fitted=tuning.WeatherAdjustedMedian().fit(train)
            pred=fitted.predict(future)
        else:
            transformer=ablation.make_transformer(model)
            xtrain=transformer.fit_transform(ablation.make_inputs(train,model))
            xfuture=transformer.transform(ablation.make_inputs(future,model))
            names=transformer.get_feature_names_out()
            estimator=sl.make_estimator('lightgbm','poisson',threads=args.threads,
                                        overrides=overrides.get('lightgbm',{}))
            estimator.fit(pd.DataFrame(xtrain,columns=names),train.carwashes.to_numpy(float))
            pred=sl.predict(estimator,'lightgbm','poisson',pd.DataFrame(xfuture,columns=names))
        p=np.asarray(pred,dtype=float)
        if len(p)!=len(future) or not np.isfinite(p).all() or (p<0).any():
            raise ValueError('Invalid future predictions')
        out=future[['issued_at','location_id','timestamp','local_date','hour','lead_day']].copy()
        out['prediction']=np.round(p,6)
        out['model']='weekly_sun__'+model
        out['forecast_type']='sunday_weekly_upcoming'
        out['created_at_utc']=datetime.now(timezone.utc).isoformat()
        all_out.append(out)
    result=pd.concat(all_out,ignore_index=True)
    if result.duplicated(['model','issued_at','location_id','timestamp']).any():
        raise RuntimeError('Duplicate future forecast hours')
    dest=Path(args.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(dest,index=False)
    print(f'Saved Sunday forecast: {dest.resolve()}')
    print(f'Mon–Sun target days: {result.local_date.min()} through {result.local_date.max()}')
    print(f'Models: {list(variants)}; sites: {result.location_id.nunique()}')
    return result


def parser():
    p=argparse.ArgumentParser(description='Generate an as-issued 7-day Sunday WashCast forecast')
    p.add_argument('--issue-date',required=True,help='Sunday YYYY-MM-DD (Chicago local date)')
    p.add_argument('--traffic',required=True)
    p.add_argument('--examples',required=True,help='Sunday-issued historical seven-day as-of examples')
    p.add_argument('--storm-features',required=True)
    p.add_argument('--recent-demand-features',required=True)
    p.add_argument('--historical-weather-csv',required=True)
    p.add_argument('--archive-csv',required=True)
    p.add_argument('--selected-configs',default='model_tuning_may_june/selected_configurations.json')
    p.add_argument('--window-days',type=int,default=365)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--models',default='lgbm_stable_recent')
    p.add_argument('--timezone',default='America/Chicago')
    p.add_argument('--closures',default=None)
    p.add_argument('--open-hour',type=int,default=8)
    p.add_argument('--sunday-open-hour',type=int,default=9)
    p.add_argument('--close-hour',type=int,default=19)
    p.add_argument('--output',required=True)
    return p


if __name__=='__main__':
    generate(parser().parse_args())
