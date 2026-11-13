WASHCAST — STORE-RELATIVE RELIABILITY + LIGHTGBM FEATURE ABLATION
===============================================================

This patch ADDS code. It does NOT overwrite previous source files, training
examples, weather data, previous predictions, or existing experiment reports.

Files to add to your existing project root:
  reliability_extended.py
  demand_feature_ablation.py

File to add to existing tests/ folder (optional):
  tests/test_demand_ablation_reliability.py

The code imports your current demand_model_benchmark.py, recent_demand_features.py,
forecast_reliability.py, storm_loss_experiment.py, model_tuning.py, and earlier
WashCast modules. No new packages are required if the previous benchmark ran.
Do not overwrite those existing modules with anything in this patch.

MOTIVATION
----------
- M5 retail forecasting: pooled LightGBM models with demand lags/rolling
  statistics and explanatory variables have been successful.
- Our earlier May-July result: recent-demand LightGBM led overall but the
  improvement over old XGBoost was concentrated at AL002; weather performance
  was mixed. We should isolate the information that truly helps.
- Recent history is essential for stores with drifting baseline demand, but
  short lags can be noisy for new stores and weather-driven periods.

1. IMMEDIATELY GET STORE-NORMALIZED RELIABILITY FROM YOUR EXISTING RESULTS
------------------------------------------------------------------------
Run from your project root in PowerShell:

.\.venv\Scripts\python.exe reliability_extended.py `
  --predictions demand_benchmark_may_july/all_predictions.csv `
  --reference weather_median `
  --comparison xgb_original `
  --output reliability_extended_may_july

Review these files:

  reliability_extended_may_july/store_relative_reliability.csv
  reliability_extended_may_july/macro_relative_reliability.csv
  reliability_extended_may_july/weather_consistency_vs_median.csv
  reliability_extended_may_july/weather_by_issuance_paired_vs_median.csv
  reliability_extended_may_july/by_observed_store_age.csv
  reliability_extended_may_july/by_historical_volume_band.csv
  reliability_extended_may_july/rainy_store_day_paired_vs_median.csv

Meaning of store WAPE: sum(abs(error))/sum(actual), expressed as %. It is
numerically MAE / mean(actual) * 100 within one site, when total actual > 0.
Thus 5 washes/hour error at a site averaging 10 washes/hour = 50% WAPE.
Do NOT use per-row MAPE because zeros and small target counts make it unstable.

Store reports include observed average washes/hour and 90th-percentile error
in washes and as a percent of that average, as well as signed bias relative
to typical traffic and paired skill vs the weather-adjusted median.
The equal-store macro-WAPE prevents higher-count / older stores from
completely dominating evaluation, although store-level sample sizes differ.

2. SHORT ABLATION CHECK (2 FORECAST ISSUANCES)
----------------------------------------------
.\.venv\Scripts\python.exe demand_feature_ablation.py `
  --examples real_results_full/asof_training_examples.csv `
  --storm-features storm_features_v1/storm_features.csv `
  --recent-demand-features recent_demand_v1/recent_demand_features.csv `
  --selected-configs model_tuning_may_june/selected_configurations.json `
  --window-days 365 `
  --validation-from 2026-05-01 `
  --validation-to 2026-07-31 `
  --validation-origins 2 `
  --threads 2 `
  --output demand_ablation_smoke

3. DEVELOPMENT FEATURE ABLATION (SAME 18 FORECAST ISSUANCES AS PREVIOUS RUN)
----------------------------------------------------------------------------
.\.venv\Scripts\python.exe demand_feature_ablation.py `
  --examples real_results_full/asof_training_examples.csv `
  --storm-features storm_features_v1/storm_features.csv `
  --recent-demand-features recent_demand_v1/recent_demand_features.csv `
  --selected-configs model_tuning_may_june/selected_configurations.json `
  --window-days 365 `
  --validation-from 2026-05-01 `
  --validation-to 2026-07-31 `
  --validation-origins 18 `
  --threads 2 `
  --output demand_ablation_may_july

Models/configurations:
  weather_median         Existing reference
  xgb_original           Existing global XGBoost + storm features (control)
  lgbm_no_recent         LightGBM without recent-demand features; isolates algorithm
  lgbm_full_recent       EXACT feature set of earlier lightgbm_recent (control)
  lgbm_stable_recent     Retains 28/56/180-day summaries, store weekday/hour
                         profiles, broad trend ratio, history/age/quality/pooled
                         profile; removes 7-day mean and exact weekly lags
  lgbm_no_short_lags     Removes only the 7/14/28-day exact-hour lags
  lgbm_no_trend_ratios   Removes the 7/56 and 28/180 trend ratios

All variants keep the SAME store ID encoding, forecast weather, storm timing,
365-day rolling training, hyperparameters, and target rows. The LightGBM
FULL variant has been verified to reproduce the original benchmark on
identical synthetic folds. Each model saves a completed fold and resumes on
an identical repeated command. Changing args requires a NEW --output folder.

Use these PowerShell commands to inspect the outputs:

Import-Csv demand_ablation_may_july/reliability/overall_metrics.csv |
  Format-Table model,mae,wape_percent,signed_bias -AutoSize

Import-Csv demand_ablation_may_july/reliability/macro_relative_reliability.csv |
  Format-Table -AutoSize

Import-Csv demand_ablation_may_july/reliability/weather_consistency_vs_median.csv |
  Format-Table model,weather_regime,issuances_with_at_least_min_hours,fraction_issuances_better,median_issuance_gain,worst_issuance_gain -AutoSize

Import-Csv demand_ablation_may_july/reliability/weather_consistency_vs_no_recent.csv |
  Format-Table -AutoSize

Import-Csv demand_ablation_may_july/reliability/store_relative_reliability.csv |
  Where-Object { $_.model -in @('lgbm_full_recent','lgbm_stable_recent','lgbm_no_recent') } |
  Format-Table model,location_id,actual_mean_washes_per_hour,mae_washes_per_hour,wape_percent,skill_vs_reference_percent -AutoSize

Import-Csv demand_ablation_may_july/reliability/by_observed_store_age.csv |
  Format-Table model,store_age_group,observations,mae,wape_percent -AutoSize

Reliability files also include paired per-issuance and per-store/issuance rain
comparisons, rainy store-day comparisons, inferred-zero label sensitivity, and
historic volume-band performance. A weather 'event' here means a FORECAST
rain-related period, not a verified storm. To attribute misses to weather
forecast errors, we need as-issued vs realized hourly weather for targets.

IMPORTANT VALIDATION RULES
--------------------------
- All historical feature rows were derived from labels available BEFORE their
  own issuance; source repeats of target hours were deduplicated.
- Training target timestamps are always strictly earlier than the test issue.
- Our 18 dates are DEVELOPMENT dates already examined, not a final holdout.
- August-September 2026 remains reserved for one frozen-model assessment.
- Closely spaced forecast issues and neighboring stores produce correlated
  errors, so the issuance-level win rate is descriptive, not a confidence bound.
- Age is based on first valid recorded traffic, NOT verified opening dates.
- For a truly never-before-seen store, the unknown store ID and little/no
  history make cold-start forecasts uncertain. A separate holdout-new-store
  experiment remains necessary before an automated deployment claim.
- WAPE is undefined if all targets are zero; the report leaves it blank.

TESTS
-----
.\.venv\Scripts\python.exe -m pytest -q tests/test_demand_ablation_reliability.py -W ignore::DeprecationWarning

PRIVACY / GIT
-------------
Add these patterns to .gitignore BEFORE running:

/demand_ablation_*/
/reliability_extended_*/

Commit only new Python scripts, optional test, and README / .gitignore.
Do not commit source CSVs, weather caches, experiment outputs, or predictions.

RESEARCH BASIS
--------------
M5 retail forecasting: https://doi.org/10.1016/j.ijforecast.2021.11.013
MLForecast lag/rolling design: https://github.com/Nixtla/mlforecast
