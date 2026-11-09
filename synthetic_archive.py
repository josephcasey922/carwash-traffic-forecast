"""Offline-only simulated archived forecasts derived from synthetic demo weather.

These mimic weather forecast errors for pipeline testing. They are NOT genuine
past forecasts and must not be used to claim real forecasting accuracy.
"""
import argparse
from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import washcast as core
from asof import issue_time


def generate(weather_path, traffic_path, path, timezone, issue_hour, seed=17):
    rng=np.random.default_rng(seed)
    weather=core.load_weather(weather_path)
    traffic=core.load_traffic(traffic_path,timezone)
    local=traffic.timestamp.dt.tz_convert(timezone)
    first=min(local.dt.date)+timedelta(days=3)
    last=max(local.dt.date)
    origin_dates=pd.date_range(first,last,freq='D')
    blocks=[]
    for origin in origin_dates.date:
        issued=issue_time(origin,timezone,issue_hour).tz_convert('UTC')
        g=weather[(weather.timestamp>issued) & (weather.timestamp<=issued+pd.Timedelta(days=9))].copy()
        if g.empty:
            continue
        lead=np.maximum(1,(g.timestamp-issued).dt.total_seconds()/86400).to_numpy()
        # Simulate forecast uncertainty growing with lead time (not leakage-safe
        # evaluation of real data: synthetic weather itself is generated first).
        g['temperature_2m'] = g.temperature_2m.to_numpy()+rng.normal(0,0.7+0.32*lead,len(g))
        g['cloud_cover'] = np.clip(g.cloud_cover.to_numpy()+rng.normal(0,5+2*lead,len(g)),0,100)
        g['precipitation']=np.clip(g.precipitation.to_numpy()*(1+rng.normal(0,0.14+0.045*lead,len(g))) +
                                   rng.normal(0,0.05+0.018*lead,len(g)),0,None)
        g['issued_at']=issued
        blocks.append(g)
    df=pd.concat(blocks,ignore_index=True)
    out=Path(path);out.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(out,index=False)
    print(f'Wrote {len(df):,} synthetic forecast-vintage rows across {len(origin_dates)} origin dates: {out}')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--weather',default='demo/weather.csv')
    parser.add_argument('--traffic',default='demo/traffic.csv')
    parser.add_argument('--output',default='demo/archive_forecasts.csv')
    parser.add_argument('--timezone',default='America/Chicago')
    parser.add_argument('--issue-hour',type=int,default=18)
    parser.add_argument('--seed',type=int,default=17)
    args=parser.parse_args()
    generate(args.weather,args.traffic,args.output,args.timezone,args.issue_hour,args.seed)
