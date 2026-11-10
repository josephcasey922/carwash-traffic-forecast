# WashCast v0.2.6 — Post-rain error explorer, weather ablations and MLP tuning

**New in v0.2.6:** an offline post-rain error explorer in the existing **Weather & MLP experiments** dashboard tab. It ranks location-days where weather-aware boosting performs worse than the historical baseline, displays the next-two-day traffic forecast against actual counts, and plots issued hourly rain and the preceding 72-hour rainfall accumulation. Optional reconstructed historical weather CSV can be shown for the preceding three days; that retrospective information is not used to fit or generate the predictions. The new helper is `rain_explorer.py`.

**Previously added in v0.2.5:** `experiments.py` and a **Weather & MLP experiments** tab in Streamlit. This version retains the existing v0.2.4 ingestion, training, forecasting, rolling-backtest, and saved-model commands unchanged. The new program is a **fully offline evaluation** of previously generated `asof_training_examples.csv` (using archived weather forecast snapshots). It never calls a weather API, does not overwrite existing models, and does not update the operational forecast model.

## Question 1: Does weather add predictive value beyond the season and calendar?

Paired ablation compares the same algorithm on the same historical test hours:

- `boosted_calendar` vs `boosted_weather`
- `mlp_calendar` vs `mlp_weather`
- `baseline` (site / weekday / hour median reference)

Calendar features include site, latitude, longitude, local hour, weekday, month, holiday/weekend, trigonometric hour/weekday, `lead_day`, and **new cyclic day-of-year features (`season_sin`, `season_cos`)**. Weather versions additionally include forecasted temperature, hourly precipitation, cloud cover, and the preceding 24h/72h rainfall known or forecast at issuance. Both weather and calendar models use identical target hours, training windows, and model-family settings. These experiments measure **predictive usefulness**, not causal weather effects.

### First experiment: quickly compare boosting with and without weather (no API)

```powershell
.\.venv\Scripts\python.exe experiments.py `
  --examples real_results_no_AL001/asof_training_examples.csv `
  --reconstructed-weather-csv weather_data/historical_weather.csv `
  --window-days 180 `
  --test-origins 8 `
  --models baseline,boosted `
  --output experiments_180d
```

`--reconstructed-weather-csv` is **optional**. It is only used **after predictions exist**, to group errors by reconstructed target-hour rain and see when an archived forecast missed significant rain. Reconstructed weather must NEVER be used as an as-of target-hour feature for the traffic forecast: this would leak future information. If you don't have this file, omit the option.

Also test a 90-day training window on the same held-out weeks:

```powershell
.\.venv\Scripts\python.exe experiments.py `
  --examples real_results_no_AL001/asof_training_examples.csv `
  --reconstructed-weather-csv weather_data/historical_weather.csv `
  --window-days 90 `
  --test-origins 8 `
  --models baseline,boosted `
  --output experiments_90d
```

### Question 2: Are rainy weeks hidden by the pooled average?

The experimental reports include:

- `overall_metrics.csv`: all-hour pooled MAE, RMSE, WAPE.
- `equal_week_metrics.csv`: **average of individual forecast-week MAEs**, so a week with more observations doesn't dominate.
- `by_issuance.csv`, `by_issue_and_regime.csv`: separate results for each forecast week, and rainy/dry regimes within it.
- `weather_gain_overall.csv`, `weather_gain_by_issuance.csv`, `weather_gain_by_regime.csv`, `weather_gain_by_month.csv`, `weather_gain_by_lead_day.csv`: **paired difference** `calendar MAE − weather MAE` on matched test hours; positive means weather helped.
- `by_weather_regime.csv`: forecasts with >=0.5 mm precipitation in target hour, otherwise >=5 mm total in previous 72 hours, otherwise dry. **Regimes use the archived forecast**, not future actual weather.
- With the optional historical weather CSV: `by_reconstructed_rain.csv`, `weather_gain_by_reconstructed_rain.csv`, `by_unforecast_rain_event.csv`, `weather_gain_by_unforecast_rain_event.csv`. These are strictly retrospective summaries of reconstructed weather, not a claim about ground-station observations.
- `by_inferred_zero.csv` and `experiment_predictions.csv`: inspect assumed zero-ticket hours and individual forecast errors.

A heavy storm not predicted by the archived weather forecast can be grouped as *forecast dry*; the retrospective reconstructed-weather output helps diagnose that. Compare **counts of examples in each group**: a very small wet sample gives noisy conclusions. Examine severe weeks separately rather than weighting them arbitrarily to win a model comparison.

### Question 3: Is the MLP trained long enough, without overfitting?

The new tuned MLP performs *nested chronological evaluation* at each rolling test origin:

1. Among the already-past outer training forecast origins, reserve the last `--mlp-validation-origins` origins for **inner validation**.
2. Fit normalization/encoders on **inner training only**; train 1 Adam epoch at a time; track training and future inner-validation MAE.
3. Use patience-based validation early stopping, selecting the best number of epochs and MLP topology/L2 penalty **without consulting the outer test week**.
4. Refit a fresh MLP on **all permissible outer training examples** for the chosen number of epochs, and evaluate on the untouched outer week.

This avoids scikit-learn's random early-stopping split, which is unsuitable for chronological forecasting. It is a development experiment and does not yet replace the MLP in `asof.py fit/forecast`.

**MLP run** (two layer-size/regularization choices, up to 300 epochs; can take appreciably longer):

```powershell
.\.venv\Scripts\python.exe experiments.py `
  --examples real_results_no_AL001/asof_training_examples.csv `
  --reconstructed-weather-csv weather_data/historical_weather.csv `
  --window-days 180 `
  --test-origins 8 `
  --models baseline,boosted,mlp `
  --mlp-candidates "12x6:0.3,24x12:1.5,32x16:0.3" `
  --mlp-epochs 300 `
  --mlp-patience 20 `
  --output experiments_180d_tuned
```

Start with `--test-origins 2 --mlp-epochs 80 --mlp-candidates "12x6:0.3"` to smoke-test the MLP before committing to all eight weeks and several candidates. The code saves `mlp_tuning.csv` and `mlp_learning_curves.csv`; Streamlit plots the learning curves per historic issuance/configuration. Beware: tune-and-compare across the same eight weeks is still **model selection**, not an untouched final test.

### October versus March: weather or another seasonal factor?

The ablation keeps calendar and cyclic day-of-year features in **both** variants so we can see whether weather improves predictions above a seasonal baseline. The feature contribution alone cannot establish causality. To evaluate both March and October, the test selection must include **held-out forecast issuances for BOTH months**, not just the last eight autumn weeks. Using the saved **weekly** archived forecast origins (no extra downloads), you can investigate March 2026 with a 90-day window:

```powershell
.\.venv\Scripts\python.exe experiments.py `
  --examples real_results_no_AL001/asof_training_examples.csv `
  --reconstructed-weather-csv weather_data/historical_weather.csv `
  --test-from 2026-03-01 --test-to 2026-03-31 `
  --test-origins 5 --window-days 90 --models baseline,boosted `
  --output experiments_march_90d
```

If there aren't enough forecast origins before the requested test dates, the script explains the problem rather than silently using future data. Weekly-only archive dates still confound lead day with weekday; full daily-origin backtesting remains a future project.

### Dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

Sidebar **Weather ablation / MLP experiments**: enter `experiments_180d` or `experiments_180d_tuned`, then open **Weather & MLP experiments**. Displays overall errors, paired weather gains, weather regimes, per-week and monthly gains, tuning choices/epoch curves, and an hourly chart of held-out predictions.

**New post-rain inspection:** open **Investigate post-rain errors · traffic and weather timeline** in the same tab. The ranked table focuses exclusively on hours tagged `Rain in previous 72h` by *archived forecast conditions*, not observed past rain. Select any listed forecast issue, location and date to see actual car washes, baseline/calendar/weather model predictions, hourly archived forecast rain, and the 72h rainfall total over the issued two-day forecast. Supply `weather_data/daily_partial/historical_weather.csv` in the optional field to overlay reconstructed historical rain going back up to 72 hours before the first target. This diagnostic never calls the weather API, does not modify model files, and does not prove a causal rain rebound.


Outputs from real data are private. `.gitignore` excludes `/experiments_*/`, `/weather_data/`, and the private-data/result folders. Review any staged git files before pushing.

---

# WashCast v0.2.4 — Offline rolling-origin backtesting

This version **does not change existing forecasting or model-training commands**. It adds `backtest.py` and a Streamlit **Rolling backtest** tab so the archived weather and traffic examples you already built can be evaluated on more than two weeks — **without any API calls**.

## Verify your inputs

Use the **one-year fit's** already-created `asof_training_examples.csv` (which contains all labeled historical forecast rows, including rows previously labeled `train`, `purged`, and `validation`). This script **ignores the old `split` designation** and creates fresh chronological train/test splits at each historical forecast issuance.

Your historical year-long weather bundle and your original traffic CSV are *not* needed for this step. The examples file already has the matched weather and actual ticket counts. Keep it **off GitHub**, as it contains company data.

### Backtest using a 180-day rolling window

```powershell
.\.venv\Scripts\python.exe backtest.py `
  --examples real_results_1yr/asof_training_examples.csv `
  --window-days 180 `
  --test-origins 8 `
  --models baseline,boosted,mlp `
  --output rolling_180d
```

### Backtest using a 90-day rolling window, on identical test weeks

```powershell
.\.venv\Scripts\python.exe backtest.py `
  --examples real_results_1yr/asof_training_examples.csv `
  --window-days 90 `
  --test-origins 8 `
  --models baseline,boosted,mlp `
  --output rolling_90d
```

Both commands use exactly the **last eight issuance dates in the input CSV**. Their test rows can therefore be compared directly. If your original one-year fit ran at weekly issue dates, the script tests eight weekly 7-day forecasts. Each fold **fits a fresh model on only past labels** from the specified number of days and then evaluates the next week's as-of archived forecast. Results are a fair comparison of rolling training windows, although not yet a final independent test.

To compare the generated metrics:

```powershell
Import-Csv rolling_180d/overall_metrics.csv | Format-Table -AutoSize
Import-Csv rolling_90d/overall_metrics.csv | Format-Table -AutoSize
```

To inspect the largest errors by week and lead day:

```powershell
Import-Csv rolling_90d/by_issue_and_lead.csv | Sort-Object {[double]$_.mae} -Descending | Select-Object -First 15 | Format-Table -AutoSize
```

Then launch the existing dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

In the sidebar, enter `rolling_180d` as **Rolling backtest results**, optionally `rolling_90d` as the **Compare** folder, then open the **Rolling backtest** tab. The dashboard checks that both comparisons contain **exactly the same target rows** before showing side-by-side MAE.

### Output files

- `backtest_predictions.csv`: actual vs. predicted hourly wash count, model, issuance date, and signed/absolute error for every fold
- `overall_metrics.csv`: MAE, RMSE, WAPE, count, signed bias per model across all tested weeks
- `by_issuance.csv`: whether particular forecast weeks were unusually difficult
- `by_issue_and_lead.csv`: which origin/lead-day pairs cause unusually large errors
- `by_lead_day.csv`, `by_location.csv`, `by_weekday.csv`: grouped diagnostics
- `by_inferred_zero.csv`: compare prediction errors for inferred-zero and recorded-hour labels
- `backtest_info.json`: chosen rolling window, historical issuance dates, and last included training target for each fold

The backtest **does not overwrite saved production models** or change the existing 6-month and 1-year fit results. Each fold refits the requested candidate models for evaluation only; it doesn't perform continuous MLP `partial_fit` updates. It retains the current model settings (no custom chronological MLP early stopping yet).

### Important limits

1. **Weekly-only issuance dates mean lead day and target weekday remain confounded.** A daily archived forecast bundle, fetched once later, would let us compare the same target date at several forecast leads.
2. **The last eight origins are not independent hourly observations.** Summarize MAE by forecast week and inspect the variability across weeks before drawing firm conclusions.
3. **Inferred zeros may represent system outages or unrecorded closures.** Check `data_quality.json` and `by_inferred_zero.csv` before trusting the scores.
4. **A full 365-day rolling model comparison over eight weeks requires enough prior archived forecast issuance dates.** With roughly one year of weekly origin data, the early backtest weeks may lack a full year of training history. Start with 90 and 180 days; download additional archived origins later if a complete 365-day window is required.
5. **This is validation for model development, not a final untouched test.** For a final accuracy claim, reserve still-later dates after choosing models and hyperparameters.

---

# WashCast v0.2.2 — schedule-aware hourly wash counts

**New operating-hour defaults:** Monday–Saturday **8:00 AM–7:00 PM** (hour starts 8 through 18) and Sunday **9:00 AM–7:00 PM** (starts 9 through 18), in `America/Chicago` by default. No training or forecast rows are produced outside those times. The schedule applies to every site until site-specific hours are implemented.

**Zero-ticket hours:** Within each site's first and last *observed* timestamp, missing operating-hour rows are now filled with `carwashes=0` and `inferred_zero=True`. Actual recorded zeros remain `inferred_zero=False`. No rows are invented before a site's first ticket or after its last ticket, and known full-day closures in `closures.csv` are excluded. The `data_quality.json` report contains counts of inferred zeros and lists completely inferred zero days. **Review any such days:** they may represent outages, closures, or incomplete exports rather than real zero traffic. Partial-day closures are not yet supported.

**Your company export works without column renaming:**

```csv
hour,location_code_id,longitude,latitude,num_carwashes
2025-02-04T11:00:00-06:00,AL001,-86.2800751,33.6110764,6
```

The `washcast.py` importer accepts both these column names and the original `timestamp,location_id,latitude,longitude,carwashes` schema. Offset-aware timestamps are converted to UTC for matching the weather, while opening-hour rules use local time.

A fresh training run is required after this preprocessing change. **Do not reuse v0.2.1 model files:** they were trained with a different schedule and zero-hour policy. For example:

```powershell
.\.venv\Scripts\python.exe asof.py fit `
  --traffic "C:\data\traffic.csv" `
  --start-date 2025-08-01 --end-date 2025-10-31 `
  --models baseline,mlp `
  --output asof_results_v022
```

These dates are only an example; use a real period covered by your export and archived weather. The command contacts the weather API unless you supply both offline weather CSVs (`--weather-csv`, `--archive-csv`); confirm authorization and licensing first. Saved training metadata now records both opening-hour settings. For offline testing, run the synthetic example below, with a new output directory.

---

# WashCast v0.2.1 — prior compatibility and incremental-update notes

**If a v0.2 forecast crashed with `InconsistentVersionWarning` or `No module named _loss`:** the bundled example models were created under scikit-learn 1.8.0 and might not load in your local scikit-learn 1.9.1. **Do not suppress that warning.** v0.2.2 defaults to forecasting with only the selected model (`--model mlp`) and checks model-version compatibility before loading. First create new saved models in **your** environment by running the demo fit below. Pretrained example `.joblib` binaries are intentionally not provided in this zip.

### Quick offline fix (Windows PowerShell, no API)

Run from the extracted project root with its virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import sklearn; print(sklearn.__version__)"
.\.venv\Scripts\python.exe asof.py fit --traffic demo/traffic.csv --weather-csv demo/weather.csv --archive-csv demo/archive_forecasts.csv --closures demo/closures.csv --start-date 2026-08-20 --end-date 2026-10-22 --origin-step 7 --validation-origins 2 --models baseline,mlp --output asof_results
.\.venv\Scripts\python.exe asof.py status --model-dir asof_results
.\.venv\Scripts\python.exe asof.py forecast --traffic demo/traffic.csv --weather-csv demo/weather.csv --archive-csv demo/archive_forecasts.csv --closures demo/closures.csv --origin-date 2026-11-08 --model-dir asof_results --model mlp --output forward_results
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

The offline example is restricted to weather archive dates included in `demo/archive_forecasts.csv`; **November 8 is a simulated issuance, not a current-day forecast**. For live forecasting with a real CSV, leave off `--weather-csv`, `--archive-csv` and `--origin-date`, and confirm an approved/licensed weather API before calling it. Use a **separate** directory from the demo models and don't put company data in a public git repository.

### Model history

Run `python asof.py status --model-dir asof_results`. In addition to the original `last_training_target_utc`, model metadata now stores `last_full_fit_at_utc`, `last_incremental_update_at_utc`, number of updates, and Python/scikit-learn/NumPy/pandas/joblib versions. The dashboard shows these fields on the Training timeline tab. Full fit resets incremental-update metadata.

Incremental learning only updates the MLP's network weights; its one-hot encoding and scaling remain frozen. New locations, substantial regime shifts, changed features, and software upgrades require a **full refit**. The updater keeps `model_training_keys.csv` so labeled records from an already seen issuance can be added later when available, without reusing previously learned rows. If no new labeled examples exist, it reports an error instead of silently retraining.

The MLP is not guaranteed to improve with every incremental update; measure holdout errors periodically and compare it with the historical baseline. `partial_fit` changes the network without rebuilding it, whereas `forecast` merely predicts and does not change weights.

Example update after later historical counts are available, using the offline example:

```powershell
.\.venv\Scripts\python.exe asof.py update --traffic demo/traffic.csv --weather-csv demo/weather.csv --archive-csv demo/archive_forecasts.csv --closures demo/closures.csv --start-date 2026-10-23 --end-date 2026-10-30 --origin-step 1 --model-dir asof_results
.\.venv\Scripts\python.exe asof.py status --model-dir asof_results
```

### Compatibility note

Joblib/pickle-based scikit-learn models are not guaranteed to load across scikit-learn releases. Even if a saved model's Python code looks the same, internal estimator types can change. **Refit after any dependency upgrade** and keep your own environment versions stable during deployment. Never load arbitrary untrusted `.joblib` files. See https://scikit-learn.org/stable/model_persistence.html .

---

# WashCast v0.2 — Hourly car-wash traffic forecasting

**Scope:** Predict *wash counts* by location and hour using prior traffic and weather. The Streamlit UI graphs the next seven days, compares five methods, and can plot an **illustrative revenue proxy** if you enter an assumed dollars-per-wash rate. Wash counts alone do not establish actual revenue, particularly if memberships are involved. No labor scheduling is implemented.

This is a **local Python prototype, not approved for staffing decisions or commercially licensed for its default public weather host**. The offline demo is wholly synthetic. No company database connection is required: provide a timestamped CSV whenever you have an export.

## 1 — Setup (Windows PowerShell)

Open a terminal in this extracted project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

The ZIP includes demo data, a simulated archive of weather *forecasts issued on past dates*, and updated example charts/results produced by all five models under the new schedule. Saved estimator `.joblib` files are intentionally excluded; fit models locally under your installed dependency versions. It doesn't require an Internet connection to inspect the dashboard.

### Launch the Streamlit app

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

Use the five tabs to explore the forecast-issuance timeline, unseen validation predictions, model behavior, daily/hourly next-week predictions, and local commands to retrain or forecast. The default sidebar paths (`example_asof`, `example_forward`) point at the saved example outputs.

## 2 — Walkthrough on fully synthetic data (offline)

The included demo contains 16 fictional sites, about 96 days of *simulated* hourly wash counts with missing records and some closures, synthetic hourly weather, and 93 daily vintage forecasts with forecast errors that increase with lead time.

```powershell
.\.venv\Scripts\python.exe asof.py fit `
  --traffic demo/traffic.csv `
  --weather-csv demo/weather.csv `
  --archive-csv demo/archive_forecasts.csv `
  --closures demo/closures.csv `
  --start-date 2026-08-20 --end-date 2026-10-27 `
  --origin-step 7 --validation-origins 2 `
  --models baseline,poisson,negbin,boosted,mlp `
  --output example_asof

.\.venv\Scripts\python.exe asof.py forecast `
  --traffic demo/traffic.csv `
  --weather-csv demo/weather.csv `
  --archive-csv demo/archive_forecasts.csv `
  --closures demo/closures.csv `
  --origin-date 2026-11-08 --model-dir example_asof `
  --revenue-per-wash 12.50 --output example_forward
```

`--revenue-per-wash 12.50` is *only a hypothetical revenue multiplier*; leave it at zero until you have a justifiable assumption. To make new synthetic demo data, run `washcast.py demo`, then `synthetic_archive.py --weather demo/weather.csv --traffic demo/traffic.csv` (which simulates forecast vintages with deliberately imperfect future weather).

## 3 — Tomorrow: one real CSV, chosen date window

Export these columns (extra columns are allowed); the company aliases shown above also work:

```csv
timestamp,location_id,latitude,longitude,carwashes
2026-11-02 09:00:00,101,33.52,-86.81,24
2026-11-02 10:00:00,101,33.52,-86.81,31
2026-11-02 09:00:00,102,33.61,-86.72,18
```

The timestamps may represent hourly counts, or individual washes (`carwashes=1` on each event). The pipeline sums multiple rows in a site/hour. Naive timestamps are interpreted using `--timezone`, so confirm UTC vs local time in your source. Use offset-aware timestamps whenever possible. **All sites must share a timezone in v0.2.** DST-ambiguous naive timestamps fail loudly rather than silently being shifted.

**Data reliability in v0.2.2:** Recorded zeros stay zero; absent operating-hour records between a site's first and last ticket become *inferred* zeros. Review the zero-fill audit before modeling, particularly completely inferred days. Provide a `closures.csv` with `location_id,date` for confirmed *entire-day* closures; site-specific/partial-day hours require later work. Keep all two years of history in the same CSV and choose which *forecast issuance dates* become training candidates:

```powershell
.\.venv\Scripts\python.exe asof.py fit `
  --traffic "C:\data\traffic.csv" `
  --start-date 2026-08-01 --end-date 2026-10-31 `
  --origin-step 7 --validation-origins 2 `
  --timezone America/Chicago --open-hour 8 --sunday-open-hour 9 --close-hour 19 `
  --output asof_results
```

If `--start-date` and `--end-date` are omitted, the system selects approximately the most recent 90 days of forecast origins through the last day with recorded hourly counts minus one day. It reads the full CSV but only asks the weather service for the time window needed for these origins, and it includes observed traffic labels and operating-hour zeros inferred within each site's recorded date range.

**The command above uses the weather API. Do not run it with work data until the API provider has approved/licensed your intended use.** The public Open-Meteo free API is non-commercial. If you have downloaded *historical observed/reconstructed* and *historical issued forecast* weather CSVs from an approved source, add `--weather-csv ... --archive-csv ...` to avoid any API request. You can set `OPEN_METEO_SINGLE_RUN_URL`, `OPEN_METEO_HISTORY_URL`, and `OPEN_METEO_API_KEY` for supported licensed endpoints; confirm endpoint entitlement and parameter compatibility with your provider.

### Estimated API requests

The current archival implementation makes roughly **sites × forecast origins** archived Single Runs requests, plus historical-weather requests per site. With 16 sites, 90 days, and `--origin-step 7`, this is approximately **208 archive requests**, cached by URL and parameters. `--origin-step 1` makes roughly 1,440 requests for the same period. This may carry cost/rate limits and take time; begin with weekly origins. The weather API's actual accessibility and your organization's license have *not* been integration-tested here.

## 4 — What is the training set and what is the test set?

For each simulated historical decision (e.g. Sunday October 18 at 6 PM local), construct rows for Monday through Sunday, for each hour and site:

- **Target (`y`)**: Actual wash count in the target site/hour *recorded later*.
- **Features (`X`)**: Site ID/coordinates, local hour/day/month/holidays, day-ahead horizon; reconstructed/available weather **before** the issuance; *archived forecasts issued before the decision* for target-hour temperature/rain/cloud and the intervening hours. Preceding 24h/72h rainfall is calculated from the stitched **as-of** weather, not actual weather from the future.
- **No future wash counts** are used as predictive features. The current baseline computes medians by site × weekday × hour from training labels, and the ML candidates learn weather/calendar associations across all locations.

The last `--validation-origins` historical issue dates form the **validation set**. All earlier-origin candidate rows that use labels on or after the first validation issuance are **removed from model training**. In this way, the *actual* wash counts used to assess validation have not been used in fitting. (The validation set repeats some target hours under different forecast issue dates; that is deliberate because we're measuring issued forecasts.) The remaining rows form the training set.

For a second final evaluation before deployment, preserve a truly untouched later period as an **independent test set**, or run forward walk-throughs over multiple new weeks. The current demo reports a chronological **validation score**, not an independent final test score.

After comparison, every model is refitted on **all labeled historical as-of examples** and saved separately. The best validation model is selected as the default forward-forecast model. New future dates have no wash-count labels, and predictions can be issued repeatedly using these saved files **without retraining**.

The last-known hourly rain observations in training are reconstructed from a historical dataset rather than a complete archival *what sensors said that minute* feed. This is a practical approximation and should be reviewed if strict historical observation latency matters. **Single Runs forecast inputs** *after the forecast origin* are fixed as-of predictions; the old v0.1 weather look-ahead has been removed from this step.

## 5 — The models

All learn from the *same* as-of rows and the *same* chronological split. Categorical site / local hour / weekday get one-hot encodings. Numerical inputs (temperature, rainfall, previous 24/72h rain, cloud cover, coordinates, month, periodic hour/weekday transforms, holiday/weekend and lead day) get training-only median imputation and standardization.

| Candidate | What it learns | Training | Incremental support in this prototype |
|---|---|---|---|
| `baseline` | Median washes for site × weekday × hour; fallback to broader site/hour medians | Grouped medians | Rebuild |
| `poisson` | Log expected number of washes as a weighted feature sum | Regularized Poisson regression | Refit |
| `negbin` | Log expected washes with overdispersed count variance | Preliminary Poisson dispersion estimate, regularized negative-binomial GLM via statsmodels | Refit |
| `boosted` | Nonlinear interactions of weather and calendar variables | Histogram gradient boosting with Poisson loss (up to 130 boosting iterations) | Refit |
| `mlp` | Nonlinear relation via a feed-forward neural net | Hidden layers `(24, 12)`, Adam, squared-error target loss, max 120 iterations | **Incrementally update** fixed network/encoding via `partial_fit` |

These are intentionally small. The MLP's training loss may not fully converge at 120 iterations; that is a tuning opportunity, not proof it was trained incorrectly. For Poisson, negative binomial, and boosted trees, `fit` rebuilds the candidate on its chosen window. In particular **do not claim that partial_fit updates a tree ensemble or Poisson GLM**; they do not support that workflow here.

## 6 — Daily forecast, weekly updates, full retraining

**Daily:** Generate tomorrow through day 7 using the saved fitted models. This does **not** refit anything:

```powershell
.\.venv\Scripts\python.exe asof.py forecast `
  --traffic "C:\data\traffic.csv" `
  --model-dir asof_results --output forward_results
```

For live API calls `asof.py forecast` uses weather from **the time the request is made**, and the closest hourly UTC issuance time in the feature table, not a fabricated historical 6 PM weather vintage. Running a task after your data refresh (e.g. each evening) is a sensible starting cadence. You can create a scheduled Windows Task Scheduler task invoking `scripts/daily_refresh.ps1` after replacing its example CSV path. **No task is created automatically** by this project.

**After a new week of actual traffic:** Update the saved MLP's weights using only historical issued forecasts whose *target wash observations have not already been used in fitting*:

```powershell
.\.venv\Scripts\python.exe asof.py update `
  --traffic "C:\data\traffic.csv" `
  --model-dir asof_results `
  --start-date 2026-11-01 --end-date 2026-11-07 `
  --origin-step 1
```

This uses the original fitted feature encoder/scaler and only `partial_fit`s the MLP. New site IDs, changes to schedules/feature definitions, structural drift, or altered training targets require a **full refit**. **The other four saved models are unchanged** by MLP updates. After an incremental update, validation scores are stale until retested. A weekly/monthly full refit is reasonable, especially over a rolling 90-day window, but it is not required just to produce new predictions.

**Full refit after operating changes:** Run `asof.py fit` again with a new `--start-date` and `--end-date` and a fresh output folder. Choose a period after the change if prior wash counts are incompatible. Compare results on untouched later periods before promoting a new version.

## 7 — Output files

`example_asof/` (or your `--output` from `fit`):

- `asof_training_examples.csv` — auditable labeled features for each issuance / target / site, including weather vintages and lead day.
- `validation_predictions.csv` — actual washes and each model's **out-of-sample issued forecast**.
- `model_comparison.csv` — validation MAE, RMSE, WAPE.
- `horizon_metrics.csv` — errors by 1–7-day lead time.
- `models/*.joblib` — saved models and preprocessing. **Do not load unknown joblib files.**
- `model_info.json` — training dates, origins, caveats, and chosen best model.
- `data_quality.json` — expected/observed scheduled-hour diagnostics, inferred-zero counts, and entirely inferred days flagged for review.

`example_forward/` (or your `--output` from `forecast`):

- `hourly_forecast.csv` — predicted washes by site, local hour, lead day, and candidate model.
- `daily_forecast.csv` — predicted total washes and optional **illustrative** revenue proxy.
- `forecast_info.json` — which weather was used and the dollar-per-wash assumption.

Prediction intervals are deliberately **not** marked as calibrated 80% intervals in v0.2. A next improvement is a rolling-origin, separate-calibration evaluation for hourly and daily intervals by site and lead time.

## 8 — Weather API provenance & limitations

- [Open-Meteo Single Runs API](https://open-meteo.com/en/docs/single-runs-api) — archived individual initialization runs, selected using `run=`, **ECMWF IFS HRES** (`models=ecmwf_ifs`). Available since March 2024 according to documentation. The prototype uses the latest **12 UTC run whose initialization preceded the planned 6 PM local issue by at least eight hours**, allowing for computation/publication lag; this is conservative, not proof of exact historical release time for every model run.
- [Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api) — reconstructed weather for *pre-issuance* periods (not to be confused with target-hour future weather).
- [Forecast API](https://open-meteo.com/en/docs) — latest forecasts for current production-style predictions.
- [Open-Meteo terms](https://open-meteo.com/en/terms) — **the public Free API is non-commercial**. Obtain a relevant paid license or organization-approved weather-data source for use at a car wash business.

Only one timezone, the Monday–Saturday/Sunday schedule common to all sites, and a whole-day closure list are supported. A forecaster trained at 6 PM local may work differently at 8 AM; investigate issuance-hour sensitivity. Historical forecast API access has **not been executed/verified from this environment**; real-world API errors, gaps, licensing, and payload differences must be checked before company use. The synthetic scores do not establish real car wash predictive accuracy.

The original `washcast.py` from v0.1 remains as an introductory, **reconstructed-weather** experiment; `asof.py` is the updated, forecast-origin-aware workflow. See `README_v0.1.md` for legacy instructions. Use `asof.py` for all serious as-of backtests.

## v0.2.3 — Download archived weather once, train as many times as needed offline

The existing `asof.py fit` calls Open-Meteo whenever `--weather-csv` and
`--archive-csv` are omitted (unless identical URL+parameters are already cached
in `.weather_cache`). `weather_bundle.py` separates the download step from training.

You **cannot** download an entire year of *as-issued* seven-day forecasts as one
continuous time series: a weekly backtest requires a distinct archived weather
model run for every site and every issue date. The script performs these calls
**once as a batch**, caches individual responses (resumable), then exports two
ordinary CSV files. Later fits are fully offline.

From the folder containing `asof.py` and `weather_bundle.py`, first download a
bundle covering the longest training experiment you plan to run. Substitute
actual dates that exist in your traffic CSV, where test targets after the last
issue date are also recorded:

```powershell
.\.venv\Scripts\python.exe weather_bundle.py `
  --traffic data/carwash_traffic.csv `
  --start-date 2025-10-21 `
  --end-date 2026-10-20 `
  --origin-step 7 `
  --output weather_data
```

Then **reuse those CSVs without contacting Open-Meteo**:

```powershell
# Full-year candidates — offline
.\.venv\Scripts\python.exe asof.py fit `
  --traffic data/carwash_traffic.csv `
  --weather-csv weather_data/historical_weather.csv `
  --archive-csv weather_data/archived_forecasts.csv `
  --start-date 2025-10-21 `
  --end-date 2026-10-20 `
  --origin-step 7 `
  --models baseline,poisson,negbin,boosted,mlp `
  --output real_results_1yr

# Six-month candidates — same issue-date cadence and final date, offline
.\.venv\Scripts\python.exe asof.py fit `
  --traffic data/carwash_traffic.csv `
  --weather-csv weather_data/historical_weather.csv `
  --archive-csv weather_data/archived_forecasts.csv `
  --start-date 2026-04-07 `
  --end-date 2026-10-20 `
  --origin-step 7 `
  --models baseline,poisson,negbin,boosted,mlp `
  --output real_results_6mo
```

**Important limitations:** the bundle contains archived weather only for the
weekly issue dates selected, so changing `--origin-step` to 1 requires
additional downloaded runs. A live forecast for tomorrow/next week also needs
a new forecast snapshot; offline training is different from having continually
updated future weather. Open-Meteo licensing requirements still apply.

The bundle includes location IDs and coordinate-derived weather data. Keep
`weather_data/` out of Git, and keep company traffic CSVs outside a public repo.
`weather_bundle.py` does not upload wash counts, only coordinates, dates, and
weather variables are sent to Open-Meteo.
