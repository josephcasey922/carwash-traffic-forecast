# WashCast — Sunday 6 PM weekly forecast add-on

This add-on is **separate from your daily wash-count model**. It builds one
seven-day **Sunday 6 PM America/Chicago** forecast for **Monday through the next
Sunday**, 76 regular operating hours per location (Mon–Sat 8 AM–7 PM, Sun
9 AM–7 PM). The daily forecast remains unchanged.

The Sunday model is a **different fitted model** using the same LightGBM feature
family and selected hyperparameters, but it learns from *Sunday-issued historical
forecasts for all seven lead days*. Simply extending a Day-1/Day-2 trained model
to days 3–7 would be extrapolation, so we do **not** do that.

A weekly forecast probably becomes less precise later in the week, especially
when the weather forecast changes. Measure lead-day 1–7 and compare with the
*matched* later daily Day-1 predictions; do not assume weekly staffing
alignment will match daily results.

## Install

Extract the ZIP into `carwash-traffic-forecast/`.

Replace these **existing** files from the ZIP:

- `washcast_validation_dashboard.py`
- `washcast_dashboard_core.py`
- `fetch_daily_archives.py` (backwards compatible: now additionally accepts
  `--forecast-days 7 --sundays-only`)

Add these **new** files:

- `apply_asof_dst_fix.py`
- `sunday_weekly_prepare.py`
- `sunday_weekly_forecast.py`

Keep the existing trained daily model and all daily prediction CSVs. The
`tests/test_sunday_weekly.py` test file is optional.

## Step 0 — fix an important daylight-saving bug

Earlier `asof.issue_time` added elapsed hours to midnight *after* localizing.
On DST-change Sundays this resulted in 5 PM or 7 PM local instead of 6 PM.
Apply the minimal, idempotent patch **before** building new archives/examples:

```powershell
.\.venv\Scripts\python.exe apply_asof_dst_fix.py
```

It alters one expression in `asof.py` or reports that it is already fixed.
Existing saved examples are not rewritten; the new weekly dataset must be
built **after** the fix. The archive downloader checks the corrected issuance
and may request missing DST-Sunday vintages.

## Step 1 — verify 7-day weather archives for Sunday issuances

```powershell
.\.venv\Scripts\python.exe fetch_daily_archives.py `
  --locations washcast_weather_grid_mapping.csv `
  --start-date 2025-09-07 `
  --end-date 2026-11-01 `
  --forecast-days 7 `
  --sundays-only `
  --output weather_data/daily_partial `
  --dry-run
```

**Important:** an earlier dry run with the default two-day horizon does not
prove that days 3–7 are present. The underlying API downloaded 10-day runs,
so most Sunday issuances should already have the additional days, but verify.
If missing, repeat without `--dry-run`. The downloader preserves existing
weather and respects its request caps; it may take several runs if much data
is missing. Re-run `--dry-run` until missing Sunday/site combinations = 0.
There is no need to request non-Sunday weekly forecast vintages.

## Step 2 — generate a NEW seven-day Sunday historical-example dataset

This reuses `asof.py` but does **not** overwrite `asof_oct_nov_refresh`.
Run from the project root, with your refreshed traffic file:

```powershell
.\.venv\Scripts\python.exe asof.py fit `
  --traffic data/carwash_traffic.csv `
  --weather-csv weather_data/daily_partial/historical_weather.csv `
  --archive-csv weather_data/daily_partial/archived_forecasts.csv `
  --start-date 2025-09-07 `
  --end-date 2026-11-01 `
  --origin-step 7 `
  --issue-hour 18 `
  --forecast-days 7 `
  --validation-origins 1 `
  --models baseline `
  --skip-missing-archive-origins `
  --output asof_sunday_weekly
```

- **September 7, 2025 and November 1, 2026 are Sundays.** This yields Sunday
  issuance histories spaced seven days apart.
- The baseline model fit inside `asof.py fit` is only a side effect of
  generating the examples. Our weekly LightGBM will be trained separately.
- Review `asof_sunday_weekly/skipped_archive_origins.csv` and make sure there
  are no unexpected missing Sunday origins. An issuance with missing archived
  weather must not be replaced with realized future weather.
- The last fully observable Sunday issue with traffic through November 10 is
  **November 1**, which targets November 2–8. A November 8 Sunday forecast
  targets November 9–15 and is not yet completely observable on November 12.

## Step 3 — generate the matching weather and recent-demand features

```powershell
.\.venv\Scripts\python.exe storm_features.py `
  --examples asof_sunday_weekly/asof_training_examples.csv `
  --historical-weather-csv weather_data/daily_partial/historical_weather.csv `
  --archive-csv weather_data/daily_partial/archived_forecasts.csv `
  --output storm_features_sunday_weekly/storm_features.csv

.\.venv\Scripts\python.exe recent_demand_features.py `
  --examples asof_sunday_weekly/asof_training_examples.csv `
  --max-inferred-per-day 3 `
  --output recent_demand_sunday_weekly/recent_demand_features.csv
```

We intentionally retain the same historic quality policy and wash-count
feature definitions. These are separate inputs from the existing daily model.

## Step 4 — backtest October–early November Sunday forecast weeks

The benchmark script's date selection uses **UTC issuance dates**. On November
1, 2026, 6 PM CST is November 2 UTC, hence the inclusive end date below.

```powershell
.\.venv\Scripts\python.exe demand_feature_ablation.py `
  --examples asof_sunday_weekly/asof_training_examples.csv `
  --storm-features storm_features_sunday_weekly/storm_features.csv `
  --recent-demand-features recent_demand_sunday_weekly/recent_demand_features.csv `
  --selected-configs model_tuning_may_june/selected_configurations.json `
  --window-days 365 `
  --validation-from 2026-10-01 `
  --validation-to 2026-11-02 `
  --validation-origins 5 `
  --models weather_median,lgbm_full_recent,lgbm_stable_recent `
  --threads 2 `
  --output sunday_weekly_backtest_oct_nov
```

This produces `sunday_weekly_backtest_oct_nov/all_predictions.csv` containing
Sunday-issued **lead days 1–7**. The original daily `lgbm_stable_recent` remains
unchanged. Do not compare raw pooled MAEs without matching target hours.

## Step 5 — prepare a nonconflicting dashboard source

```powershell
.\.venv\Scripts\python.exe sunday_weekly_prepare.py `
  --predictions sunday_weekly_backtest_oct_nov/all_predictions.csv `
  --output sunday_weekly_oct_nov/weekly_predictions.csv
```

The converter validates that each forecast issued on **Sunday at 6 PM Chicago
time**, that target days really are lead 1–7, and reports incomplete store-weeks
in `sunday_weekly_oct_nov/weekly_coverage.csv`. It namespaces weekly models as
`weekly_sun__lgbm_stable_recent`, etc., to distinguish them from nightly
predictions. This file is automatically discovered by the dashboard.

## Step 6 — view the weekly forecast and the daily comparison

```powershell
.\.venv\Scripts\python.exe -m streamlit run washcast_validation_dashboard.py
```

Keep the normal daily `demand_ablation_oct_nov_refreshed/all_predictions.csv`
selected, and ALSO select `sunday_weekly_oct_nov/weekly_predictions.csv` in
**Saved experiment CSVs**.

- **Sunday weekly forecast tab:** choose a store and Sunday issue. See Monday
  through Sunday hourly charts, one row for each lead day, Day-1 daily-vintage
  overlay, paired hourly MAE and Forecast Labor Alignment.
- **Hourly day / Seven-day view:** check **Overlay Sunday forecast on day/week
  charts** in the sidebar. Purple dashed lines are the Sunday-issued forecast;
  blue remains the usual daily-updated forecast.
- The week view uses the week starting **Monday**; the origin is the Sunday
  immediately before it.
- Historical weekly metrics only describe **available observed hours**. A
  partial week is flagged; do not claim full-week accuracy on incomplete data.
- Labor-alignment is a secondary diagnostic, not the model's training target.

## Optional: issue an actual seven-day forecast

Once an as-issued Sunday weather run is archived, this command can create an
unscored next-week forecast. **Do not use its output as an accuracy report.**
For example, to replay the Sunday **November 8, 2026** issuance (this date is
already in the past), with traffic labels restricted chronologically:

```powershell
.\.venv\Scripts\python.exe sunday_weekly_forecast.py `
  --issue-date 2026-11-08 `
  --traffic data/carwash_traffic.csv `
  --examples asof_sunday_weekly/asof_training_examples.csv `
  --storm-features storm_features_sunday_weekly/storm_features.csv `
  --recent-demand-features recent_demand_sunday_weekly/recent_demand_features.csv `
  --historical-weather-csv weather_data/daily_partial/historical_weather.csv `
  --archive-csv weather_data/daily_partial/archived_forecasts.csv `
  --selected-configs model_tuning_may_june/selected_configurations.json `
  --models lgbm_stable_recent `
  --window-days 365 `
  --threads 2 `
  --output sunday_weekly_live/upcoming_weekly_forecast_20261108.csv
```

The forecast is explicitly unscored: its CSV includes **prediction**, and
**no `carwashes` column**. The dashboard's sidebar has an **Upcoming Sunday
forecast** selector, and its weekly tab offers an unscored forecast mode. This
safeguard avoids interpreting unavailable actuals as zeros.

For a truly new Sunday schedule, refresh the as-of historical examples, past
weather and corresponding archived issuance forecast available **at that
Sunday 6 PM** first. The tool uses the selected Sunday's forecast vintage; it
does not query a live weather API or automatically schedule itself. The user
should only publish the forecast after the chosen issue time.

### Important limitations

1. The seven-day model has not yet been evaluated on your refreshed private
   data here; expect larger errors for days 4–7. Inspect by lead day, not just
   overall MAE.
2. Existing historical closures/holidays (for example early closing on July 4)
   remain governed by the source operating calendar. They are not silently
   cleaned by this patch.
3. `asof.py fit` and weather archives are separate data-preparation operations;
   always keep the original daily model artifacts and weekly files separate.
4. Daily Day-1 forecasts are issued **later** than Sunday weekly forecasts for
   most target days. Comparing them on matched target hours measures the value
   of fresher information and possibly different fitted models, not solely the
   quality of LightGBM itself.
5. The current weekly forecast script fits on historical Sunday issues for
   the past 365 days. It does not create a persistent saved LightGBM model, nor
   install automatic scheduling; it is safe to run explicitly each Sunday.
6. We intentionally preserve historical missingness and score official
   observations; do not optimize weekly results by silently dropping unusual
   events or inferred zeros.

## Files in this patch

- `washcast_validation_dashboard.py` — updated Streamlit viewer
- `washcast_dashboard_core.py` — tested pure schema, pairing, alignment helpers
- `fetch_daily_archives.py` — optional 7-day Sunday weather archive auditing
- `apply_asof_dst_fix.py` — surgical one-line patch for `asof.py`
- `sunday_weekly_prepare.py` — namespace and validate historical Sunday forecasts
- `sunday_weekly_forecast.py` — Sunday 6 PM as-issued model fit/predict
- `tests/test_sunday_weekly.py` — calendar, DST, pairing, scoring tests
- `tests/test_washcast_dashboard_core.py` — existing dashboard tests
