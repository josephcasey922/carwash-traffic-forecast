"""Build a self-contained offline HTML report from the synthetic demo outputs."""
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

HERE=Path(__file__).parent
v=pd.read_csv(HERE/'example_asof/validation_predictions.csv')
compare=pd.read_csv(HERE/'example_asof/model_comparison.csv')
daily=pd.read_csv(HERE/'example_forward/daily_forecast.csv')
info=json.loads((HERE/'example_asof/model_info.json').read_text())
rows=v[(v.issued_at==v.issued_at.iloc[0])&(v.location_id==100)&(v.local_date==v.local_date.iloc[0])].sort_values('hour')
fig_val=go.Figure()
fig_val.add_scatter(x=rows.hour,y=rows.carwashes,name='Actual washes',mode='lines+markers',line={'width':4})
for model in ['baseline','poisson','negbin','boosted','mlp']:
    fig_val.add_scatter(x=rows.hour,y=rows[f'pred_{model}'],name=model,mode='lines+markers')
fig_val.update_layout(title='Next-day hourly validation: fictitious site 100, October 16',xaxis_title='Local hour',yaxis_title='Washes')
fig_compare=px.bar(compare,x='model',y='mae',text='mae',title='Out-of-sample MAE: fictitious data (lower is better)')
site_daily=daily[daily.location_id==100].copy()
fig_daily=go.Figure()
for m in ['baseline','poisson','negbin','boosted','mlp']:
    fig_daily.add_scatter(x=site_daily.local_date,y=site_daily[f'pred_{m}'],name=m,mode='lines+markers')
fig_daily.update_layout(title='Seven-day forecast: fictitious site 100, Nov 9–15 (scheduled hours only)',xaxis_title='Target day',yaxis_title='Predicted washes')
parts=[]
for f in (fig_val,fig_compare,fig_daily):
    parts.append(f.to_html(full_html=False,include_plotlyjs=False,config={'responsive':True,'displaylogo':False}))
html='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WashCast v0.2.2 — Offline Example</title><style>body{font-family:system-ui,-apple-system,sans-serif;margin:0;color:#172334;background:#f6f8fb}
main{max-width:1050px;margin:auto;padding:32px 22px}.panel{background:white;padding:24px;border:1px solid #e2e8f0;border-radius:16px;margin:22px 0}
h1{font-size:32px;margin:0}h2{font-size:20px}p{line-height:1.65}small{color:#59697c}code{background:#e6ecf3;padding:2px 5px;border-radius:4px}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.step{background:#eaf0f7;border-radius:10px;padding:16px}
@media(max-width:700px){.steps{grid-template-columns:1fr}}</style><script>'''+get_plotlyjs()+'''</script></head><body><main>
<h1>WashCast v0.2.2 · Backtest & Forecast</h1><p><b>Entirely synthetic example.</b> No observed company traffic or real archived weather is represented in these charts. Operating hours: Mon–Sat 8 AM–7 PM, Sunday 9 AM–7 PM. Hours with no synthetic ticket record inside each site's observed period are labeled inferred zeros.</p>
<div class="panel"><h2>How the model trained for the following day</h2><div class="steps">
<div class="step"><b>1. On Oct 15 at 6 PM</b><p>Known historical traffic and weather up through issuance.</p></div>
<div class="step"><b>2. Look at issued forecasts</b><p>Archived simulated forecast weather for Oct 16–22, plus location and calendar.</p></div>
<div class="step"><b>3. Predict and validate</b><p>Estimate hourly washes. Evaluate against actual synthetic counts recorded later.</p></div></div>
<p>Earlier issuance dates form the training set. Validation issuance dates are <code>Oct 15</code> and <code>Oct 22</code>. Any earlier-issued training row whose target overlaps the validation period is excluded from fitting.</p></div>
<div class="panel">'''+parts[0]+'''</div><div class="panel">'''+parts[1]+'''</div><div class="panel">'''+parts[2]+'''</div>
<div class="panel"><h2>Model refresh strategy</h2><p><b>Daily:</b> use saved parameters with updated live weather; no retraining. <b>After new observations:</b> optionally partial-fit the MLP with newly labeled examples. <b>Periodic/structural changes:</b> full chronological refit of every model and reassess on a separate holdout.</p>
<small>These are point forecasts. No validated prediction intervals or real revenue are included. Revenue proxy in Streamlit is counts × assumed dollars per wash, not actual recognized revenue.</small></div>
</main></body></html>'''
output=HERE/'example_report.html'
output.write_text(html,encoding='utf-8')
print(f'{output} ({len(html)/1024/1024:.1f} MiB)')
