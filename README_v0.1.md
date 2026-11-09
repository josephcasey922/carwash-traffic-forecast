# Car Wash Hourly Traffic Forecast — prototype v0.1

This is a **prototype** to practice data ingestion and model comparison. It forecasts washes, **not staff**, using an hourly traffic CSV, coordinates, and either supplied weather or an Open-Meteo API request. It is not a production-ready forecast.

## 1. Install (Windows PowerShell)

```powershell
cd carwash-traffic-forecast
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell restricts activation, run `.\.venv\Scripts\python.exe` in place of `python` in the commands.

## 2. Test with wholly synthetic data (no internet, no company files)

```powershell
python washcast.py demo
python washcast.py run --traffic demo/traffic.csv --weather-csv demo/weather.csv --closures demo/closures.csv --forecast-start 2026-11-09 --output demo_results
python -m pytest -q
```

The demo produces 16 **fictitious** locations; 96 days of hourly washes, with random data gaps and some closure dates; synthetic hourly weather for the history and following eight days. Counts depend on **hour, day of week, temperature, current rain, and preceding rain**. It is synthetic, not historical weather or actual company traffic. Expect noisy model scores rather than a guaranteed winner.

To save time on the first run, use `--models baseline,poisson,boosted`, then try all five.

## 3. Swap in a real CSV (workplace test)

Example `traffic.csv`:

```csv
timestamp,location_id,latitude,longitude,carwashes
2026-11-02 09:00:00,101,33.52,-86.81,24
2026-11-02 10:00:00,101,33.52,-86.81,31
2026-11-02 09:00:00,102,33.61,-86.72,18
```

The columns must be named exactly as shown. `timestamp` can be an ISO timestamp including `Z`/UTC offset, or a timestamp with **no** zone interpreted using `--timezone` (same local timezone for all sites in this prototype). If the sites span multiple timezones, split exports by zone for now. DST-ambiguous local timestamps are rejected; UTC with `Z` is recommended.

`carwashes` must be a nonnegative whole-number count. One row per hour works; multiple records per hour are **summed**, so per-transaction exports with `carwashes=1` also work. Site coordinates must remain stable. If all you have is a daily count, this program is *not* sufficient: it needs hourly timestamps.

For a personal prototype, if API use is authorized:

```powershell
python washcast.py run --traffic "C:\workdata\traffic.csv" --timezone America/Chicago --train-days 90 --validation-days 14 --open-hour 7 --close-hour 19 --forecast-days 7 --output real_test
```

This makes API requests using the site coordinates; caches responses locally. It needs an internet connection. **No actual request was run during creation of this starter project.** Check API usage terms for company use (see below).

For testing with a previously downloaded weather CSV, replace API access with `--weather-csv path/to/weather.csv`.

`--forecast-start` defaults to *tomorrow* based on the specified timezone. During demos, the explicit date ensures all future weather rows exist in the synthetic dataset.

### Opening hours, closures, and missing records

- Hours in `--open-hour` (inclusive) to `--close-hour` (exclusive) are used for training and forecasting. v0.1 applies one common schedule to all sites.
- A **missing hourly row is unknown**, never automatically a zero; reported missing hours are excluded from model fitting.
- A recorded `carwashes=0` is a real zero and **is** used in training.
- Optional `closures.csv` columns are `location_id,date` (e.g., `101,2026-11-26`). It excludes these *entire local dates* from both training and forecasts. Partial-day closures are not yet supported.
- Without a closure log, a historically closed day that has no rows won't be learned as zero. A future holiday/closure must be excluded using the closure file. The built-in US federal holiday indicator is **not** an operating-hours schedule.
- If an export only includes transactions and drops open hours with truly zero sales, there is **no reliable way to distinguish those zeros from missing data**. Obtain a continuous hourly count query or an open-hours/closure indicator before trusting evaluation.
- `data_quality.json` reports missing *nominal* scheduled hours and the count of labeled rows.

### Weather CSV

A weather CSV used for offline training/forecasting needs:

```csv
timestamp,location_id,temperature_2m,precipitation,cloud_cover
2026-11-02T09:00:00+00:00,101,18.5,0.0,21
```

- `timestamp`: UTC or timezone-aware ISO-8601 preferred.
- `temperature_2m`: °C, `precipitation`: mm per hour, `cloud_cover`: percent.
- Include **each hour** for history **and** future seven days, plus at least three days prior to the earliest traffic record for full lag features. There can be gaps, but missing weather is imputed during training; **future target weather is required**.
- Each location is joined to weather on `(location_id, UTC hour)`.

## 4. Outputs

`results/` contains:

| File | Use |
|---|---|
| `merged_training_data.csv` | Inspect each traffic hour after its weather join, rainfall lags, and calendar feature construction |
| `model_comparison.csv` | Baseline, Poisson, negative-binomial GLM, gradient-boosted Poisson trees, and tiny MLP: MAE, RMSE, WAPE on a **chronological** holdout |
| `validation_predictions.csv` | Hour-level true versus forecast for each trained model |
| `hourly_forecast.csv` | One row per future location-hour; predicted washes, exploratory bands and weather features |
| `daily_forecast.csv` | Expected daily wash totals (bounds are **sums of per-hour bounds, not validated daily prediction intervals**) |
| `data_quality.json` | Per-site missing-hour diagnostics |
| `run_summary.json` | Selected model and essential limitations |
| `best_model.joblib` | Locally saved fitted model; only load joblib files you trust |

The model with lowest validation MAE is refitted to all supplied labeled data before future forecasting. This provides a simple consistent comparison, not a definitive test of deployment accuracy. One common model is learned across sites; location identity is included as a categorical feature.

## 5. How the models are trained

1. Take the latest `--train-days` of traffic data (default 90).
2. Aggregate to exact hourly UTC timestamps, preserve **observed** zeros, never synthesize missing counts.
3. Attach each site's historical hourly weather and calculate preceding 24h and 72h rainfall. Weather lag calculations *exclude the target hour*.
4. Add location, local hour, weekday, month, federal-holiday indicator, site coordinates, and weather features.
5. Hold out the last 14 **local calendar dates** to compare candidate models; train only on earlier dates.
6. Refit the winning model on all available labeled hours and issue future predictions for the next seven complete local dates.

Training uses historical **realized or reconstructed** weather, while deployment uses **forecasted** weather. This is useful to verify joins and measure weather relationships, but **the validation numbers are not a true as-of-weather forecast backtest**. Even with a chronological traffic split, future actual weather leaks into the test feature data relative to when a real forecast would have been issued. Forecast errors can therefore be materially worse in deployment. Later replace historical realized weather features with Open-Meteo archived **previous runs/single runs** corresponding to the exact forecast issuance date; train/evaluate separately by prediction horizon.

### Prediction intervals are exploratory

`lower_80_approx` and `upper_80_approx` are symmetric, count-scaled bands based on the final holdout's absolute errors. They are **not reliably calibrated 80% predictive intervals** because (i) model selection and calibration share a holdout, (ii) weather forecast errors aren't reproduced historically, and (iii) coverage varies by site, hour, and forecast horizon. We should replace them with rolling-origin calibrated intervals later. Do not use them for operational staffing decisions yet.

### Important forecast limitations

- The initial model predicts each future hour directly, **without traffic lags for target hours that wouldn't be observed yet**. It still learns historical demand patterns from the training labels. Current-day and earlier weather influence predictions via rainfall lags, and for later future days this lag naturally includes *forecast* rain from earlier future days.
- There is no use of weather on *days after* the target hour.
- Daylight saving time is managed at the timestamp level by converting hourly UTC records to the configured local timezone.
- The model has no per-site operating hours, vacation/holiday business calendar, promotion or construction information, special events, or drift-detection/retraining logic yet.
- Only one time zone across all sites is supported; split sites if their time zones differ.
- No SQL credentials, database dependencies, or company information should be committed to a public repository.

## 6. Weather providers and API rights

Default URLs are `https://historical-forecast-api.open-meteo.com/v1/forecast` (past conditions) and `https://api.open-meteo.com/v1/forecast` (upcoming weather). To use licensed endpoints or a key, set `OPEN_METEO_API_KEY` and `OPEN_METEO_HISTORY_URL` / `OPEN_METEO_LIVE_URL` as environment variables. The endpoints and key must be those provided by your weather-data license.

**Open-Meteo's free hosted API is designated for non-commercial use.** Do not assume it is permitted for internal company operations or company data processing; confirm licensing/approval or use an approved provider before workplace deployment or an actual work-data API run.

Weather sources:
- https://open-meteo.com/en/docs
- https://open-meteo.com/en/docs/historical-forecast-api
- https://open-meteo.com/en/docs/previous-runs-api
- https://open-meteo.com/en/terms

## 7. Next improvements after you test real data

- Confirm whether records missing on an operating hour represent zero or an extraction gap, and verify site operating times.
- Compare 30/60/90/180-day training windows and consider concept drift.
- Replace retrospective actual weather with **archived weather forecasts from each historical forecast origin** for next-day / 2-7-day rolling backtests.
- Make hourly traffic predictions probabilistic and forecast-horizon-specific (better calibrated intervals).
- Support site-specific time zones and opening hours, partial-day closures, and feature-based weather-recovery dynamics.
- Start a daily retraining / reporting process only after the data mapping and evaluation are verified.
