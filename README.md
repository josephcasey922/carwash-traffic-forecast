# WashCast v0.2.1 — compatibility and incremental updates

**If a v0.2 forecast crashed with `InconsistentVersionWarning` or `No module named _loss`:** the bundled example models were created under scikit-learn 1.8.0 and might not load in your local scikit-learn 1.9.1. **Do not suppress that warning.** v0.2.1 defaults to forecasting with only the selected model (`--model mlp`) and checks model-version compatibility before loading. First create new saved models in **your** environment by running the demo fit below. Pretrained example `.joblib` binaries are intentionally not provided in this zip.

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

The ZIP includes demo data, a simulated archive of weather *forecasts issued on past dates*, and example results already produced by all five models. It doesn't require an Internet connection to inspect the dashboard.

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

Export **exactly** these columns (extra columns are allowed):

```csv
timestamp,location_id,latitude,longitude,carwashes
2026-11-02 09:00:00,101,33.52,-86.81,24
2026-11-02 10:00:00,101,33.52,-86.81,31
2026-11-02 09:00:00,102,33.61,-86.72,18
```

The timestamps may represent hourly counts, or individual washes (`carwashes=1` on each event). The pipeline sums multiple rows in a site/hour. Naive timestamps are interpreted using `--timezone`, so confirm UTC vs local time in your source. Use offset-aware timestamps whenever possible. **All sites must share a timezone in v0.2.** DST-ambiguous naive timestamps fail loudly rather than silently being shifted.

**Data reliability:** Recorded zeros stay zero. Missing expected hourly records are **unknown**, not zero. Provide a `closures.csv` with `location_id,date` for confirmed *entire-day* closures; site-specific/partial-day hours require later work. Keep all two years of history in the same CSV and choose which *forecast issuance dates* become training candidates:

```powershell
.\.venv\Scripts\python.exe asof.py fit `
  --traffic "C:\data\traffic.csv" `
  --start-date 2026-08-01 --end-date 2026-10-31 `
  --origin-step 7 --validation-origins 2 `
  --timezone America/Chicago --open-hour 7 --close-hour 19 `
  --output asof_results
```

If `--start-date` and `--end-date` are omitted, the system selects approximately the most recent 90 days of forecast origins through the last day with recorded hourly counts minus one day. It reads the full CSV but only asks the weather service for the time window needed for these origins, and it only includes observed traffic labels in that window.

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
- `data_quality.json` — missing nominal scheduled-hour diagnostics.

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

Only one timezone, one nominal set of hours for all locations, and a whole-day closure list are supported. A forecaster trained at 6 PM local may work differently at 8 AM; investigate issuance-hour sensitivity. Historical forecast API access has **not been executed/verified from this environment**; real-world API errors, gaps, licensing, and payload differences must be checked before company use. The synthetic scores do not establish real car wash predictive accuracy.

The original `washcast.py` from v0.1 remains as an introductory, **reconstructed-weather** experiment; `asof.py` is the updated, forecast-origin-aware workflow. See `README_v0.1.md` for legacy instructions. Use `asof.py` for all serious as-of backtests.
