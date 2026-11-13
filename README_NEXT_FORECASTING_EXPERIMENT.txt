WASHCAST — RECENT DEMAND / STORE ADAPTATION EXPERIMENT
====================================================

This patch ADDS the following files to your existing repository. It does not
replace washcast.py, asof.py, any saved model, or your weather data.

  recent_demand_features.py
  demand_model_benchmark.py
  forecast_reliability.py
  tests/test_recent_demand_benchmark.py

No additional pip packages are needed if your previous XGBoost/LightGBM
experiments work. MLForecast is not required in this first experiment: the
rolling/lag strategy is inspired by the M5 and MLForecast approaches, but the
issuance-specific weather keys require stricter as-of handling than a generic
single-time-series cross-validation call.

PREREQUISITES (from earlier WashCast experiments)
-----------------------------------------------
  real_results_full/asof_training_examples.csv
  storm_features_v1/storm_features.csv
  model_tuning_may_june/selected_configurations.json  (optional, recommended)

Source code modules previously installed in project root:
  backtest.py, experiments.py, washcast.py, asof.py, storm_features.py,
  storm_loss_experiment.py, model_benchmark.py, model_tuning.py

STEP 1 — CREATE THE HISTORICAL FEATURE CACHE (NO WEATHER API)
------------------------------------------------------------
From your project root, in PowerShell:

.\.venv\Scripts\python.exe recent_demand_features.py `
  --examples real_results_full/asof_training_examples.csv `
  --output recent_demand_v1/recent_demand_features.csv

This builds per-store rolling demand for 7/28/56/180 days, the 7/56 and
28/180 demand ratios, 7/14/28-day comparable-hour lags, last 28-day hourly
medians, last 56-day comparable weekday/hour medians, recent data quality,
location history and cold-start/pooled forecasts. Source data is not modified.

Historical labels are deduplicated before any rolling calculation. The source
has overlapping two-day forecast examples; otherwise one observation could be
counted twice. Only COMPLETED local days prior to the actual historical
forecast issuance are used. No unobserved store-day is assumed zero. Store-
days with unexpectedly missing hours or >3 inferred-zero hours are excluded
from the daily means; this threshold is configurable. Lag features and medians
are computed from the same local-time historical frame.

The generator was tested against the uploaded real examples: 168,283 forecast
rows, 84,393 unique store-hours, 7,521 valid store-days for daily statistics.
AL002 issuance 2026-05-13 had 7/28/56-day means of 199.3 / 221.7 / 284.0.

STEP 2 — A SHORT SMOKE RUN (TWO ISSUANCES)
----------------------------------------
.\.venv\Scripts\python.exe demand_model_benchmark.py `
  --examples real_results_full/asof_training_examples.csv `
  --storm-features storm_features_v1/storm_features.csv `
  --recent-demand-features recent_demand_v1/recent_demand_features.csv `
  --selected-configs model_tuning_may_june/selected_configurations.json `
  --window-days 365 `
  --validation-from 2026-05-01 `
  --validation-to 2026-07-31 `
  --validation-origins 2 `
  --models weather_median,adaptive_median,xgb_original,xgb_recent `
  --threads 2 `
  --output demand_benchmark_smoke

(The --selected-configs argument is optional; without it, model defaults are
used. The exact run conditions will be recorded in experiment_manifest.json.)

STEP 3 — DEVELOPMENT COMPARISON (18 CHRONOLOGICALLY SPACED ISSUANCES)
-------------------------------------------------------------------
.\.venv\Scripts\python.exe demand_model_benchmark.py `
  --examples real_results_full/asof_training_examples.csv `
  --storm-features storm_features_v1/storm_features.csv `
  --recent-demand-features recent_demand_v1/recent_demand_features.csv `
  --selected-configs model_tuning_may_june/selected_configurations.json `
  --window-days 365 `
  --validation-from 2026-05-01 `
  --validation-to 2026-07-31 `
  --validation-origins 18 `
  --models weather_median,adaptive_median,xgb_original,xgb_recent,lightgbm_recent,poisson_recent `
  --threads 2 `
  --output demand_benchmark_may_july

Each model uses identical issuance/store/target hours. For every held-out
issuance, its training labels (and original training example issuances) must
precede that issuance. Input feature columns are calculated at the original
training row's own issuance. All models use the original as-issued archived
weather and previously generated leakage-checked storm features.

The model variants:
  weather_median: previous pooled + location-adjusted weather median
  adaptive_median: same with capped, support-shrunk recent 28/180-day factor
  xgb_original: unchanged tuned XGBoost Poisson + storm predictors
  xgb_recent: same XGBoost + recent-demand inputs (controlled ablation)
  lightgbm_recent: LightGBM Poisson + same recent-demand inputs
  poisson_recent: regularized Poisson GLM with store x weather-regime terms

The small linear model is L2-regularized and includes store-specific fixed
weather interactions. It is NOT a fully Bayesian hierarchical random-effects
model. A new location with no historical traffic cannot be reliably volume-
calibrated without more covariates; its features explicitly flag cold-start.

Model runs resume from completed fold/model CSVs if you repeat the IDENTICAL
command and output directory. Changing settings requires a new output folder.

STEP 4 — INSPECT WHAT MATTERS FOR RELIABILITY
--------------------------------------------
Import-Csv demand_benchmark_may_july/reliability/overall_metrics.csv |
  Format-Table -AutoSize

Import-Csv demand_benchmark_may_july/reliability/store_macro_metrics.csv |
  Format-Table -AutoSize

Import-Csv demand_benchmark_may_july/reliability/skill_vs_weather_by_regime.csv |
  Format-Table -AutoSize

Import-Csv demand_benchmark_may_july/reliability/skill_vs_weather_by_location.csv |
  Format-Table -AutoSize

Import-Csv demand_benchmark_may_july/reliability/worst_location_issuances.csv |
  Where-Object { $_.location_id -eq 'AL002' } |
  Format-Table -AutoSize

Other reports: by_store_age.csv, by_history_support.csv, by_inferred_zero.csv,
excluding_inferred_zeros.csv, daily_total_metrics.csv, rainy_store_day.csv,
rainy_issue.csv, skill_vs_weather_rainy_store_day.csv, and raw predictions.

POSITIVE skill_percent indicates MAE lower than the weather-adjusted median
on EXACTLY the same set of forecasted target rows. Both total/hourly scores and
macro average across stores are included. The event reports cover forecast-
rainy store-days, not observed independent storms. The original source does
not contain realized weather or confirmed closure labels; those are separate
future data-quality and event-study tasks.

OPTIONAL — REUSE THE REPORTER WITH AN OLDER PREDICTION CSV
----------------------------------------------------------
.\.venv\Scripts\python.exe forecast_reliability.py `
  --predictions location_model_AL002_may_june/all_predictions.csv `
  --output forecast_reliability_AL002

TESTS
-----
.\.venv\Scripts\python.exe -m pytest -q tests/test_recent_demand_benchmark.py

Tests cover duplicate-label consistency, future-label tampering, absent days,
cold-start stores, paired error comparisons, end-to-end estimator fitting and
resuming an interrupted benchmark. Tested locally with synthetic data.

PRIVACY / GIT
-------------
Add to your .gitignore before running:

/recent_demand_*/
/demand_benchmark_*/
/forecast_reliability_*/

Only commit the FOUR code/test files and your .gitignore. Do not commit source
CSV files, cache outputs, actual forecasts, or company-derived metrics.

IMPORTANT SCOPE / EVALUATION NOTES
---------------------------------
May, June, and July are already development periods; DO NOT treat this run as
a fresh holdout. Reserve August–September 2026 to evaluate a frozen selected
configuration ONCE. Repeated forecast issuances may share labeled target hours;
site-hour errors are correlated across locations and weather systems.

If the new models improve overall MAE but worsen rainy forecasts or the worst
stores, do not replace the current model. Prioritize stable storm-event MAE,
per-store skill, data quality, worst 90th-percentile error, bias and new-store
age effects over a few hundredths of average MAE.

Research basis:
- M5 retail forecasting competition: https://doi.org/10.1016/j.ijforecast.2021.11.013
- MLForecast's efficient lag/rolling features: https://github.com/Nixtla/mlforecast
- Global forecasting and heterogeneous time series:
  https://w.robjhyndman.com/publications/global-forecasting/
