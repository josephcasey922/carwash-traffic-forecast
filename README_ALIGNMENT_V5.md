# WashCast dashboard v5 — model-labelled employee-hour alignment

This is a **dashboard-only replacement** for the Sunday weekly dashboard v4. It does not retrain or alter your carwash forecasting model, weather archives, or historical predictions.

## Install

1. Stop the Streamlit server (`Ctrl+C`).
2. Extract `washcast-dashboard-model-alignment-v5.zip` into your existing project root.
3. **Replace only** `washcast_validation_dashboard.py` and `washcast_dashboard_core.py`. Keep your `sunday_weekly_prepare.py`, `sunday_weekly_forecast.py`, `asof.py`, and existing data/CSV files unchanged.
4. Restart:

```powershell
.\.venv\Scripts\python.exe -m streamlit run washcast_validation_dashboard.py
```

Load daily `demand_ablation_*/all_predictions.csv` and the Sunday `sunday_weekly_*/weekly_predictions.csv` in the sidebar. Select your **Primary model**, **Compare with** model, and optionally **Overlay Sunday forecast on day/week charts**. The **Sunday weekly forecast** tab also offers a daily-refreshed comparison.

## Exact calculation

The dashboard maps each hourly **carwash count** to **recommended employees**:

- 0–17 washes: 2 employees
- 18–40 washes: 3 employees
- 41–60 washes: 4 employees
- 61+ washes: 5 employees

Historical forecasted counts are rounded to the nearest nonnegative integer, exact half values upward, consistent with `bucket_alignment_report.py`.

Let `R_h` = employees recommended by *actual* washes during hour h, and `P_h` = employees that would have been recommended had the store staffed **exactly according to the chosen WashCast forecast**. The hypothetical Forecast Labor Alignment percentage is:

`100 * (1 - sum_h(abs(P_h - R_h)) / sum_h(R_h))`

This exactly matches the **structure** of the company's efficiency formula. Because real scheduled employee counts are not available to the dashboard, `P_h` is forecast-implied staffing rather than historical staffing. It is a counterfactual rollout score, **not actual site labor efficiency**, and does not change the wash-count forecasting objective.

For a full Monday–Sunday week, sum the *hourly* employee differences and actual recommendations across the **entire week** (76 normal operating hours per store: six 11-hour days and one 10-hour Sunday). **Do not average seven daily percentages.** Network aggregation similarly uses all scored store-hours, not an average of store percentages.

An incomplete selected week remains a **partial-hours diagnostic** and is labelled as such. For a fair model comparison, hours are intersected and scored on the same actual observations. Daily model comparisons match issuance/store/target/lead; fixed Sunday versus daily forecasts match target store-hour only because issuance times differ. Any conflicting actual labels produce an error instead of being silently combined.

## What's new

- **Day chart:** selected daily model, optional comparison daily model, and optional fixed Sunday model are explicitly named in chart legends. Under the hourly graph, the model-by-model employee-hour alignment values, bar graph, and supporting numerator/denominator are shown on the *same matched hours*.
- **Seven-day view:** model-labelled full-week alignment values plus **alignment by individual day**. The full-week score is computed from pooled hourly employee counts—not by averaging that daily trend.
- **Sunday weekly forecast:** identifies the fixed Sunday-issued model, and on matching store-hours compares it against one or two selected nightly Day-1 models. Includes model-labelled daily alignment trend and paired daily-versus-weekly table.
- **Across months:** retains the existing monthly chart with model names and adds a matched-hours alignment comparison when the Sunday overlay is enabled.
- **Models & buckets / main KPIs:** metric labels identify the primary model rather than saying only “Labor alignment.”

## Tests

`tests/test_alignment_v5.py` and `tests/test_alignment_charts_v5.py` are optional automated tests. With the v4 dependencies present, run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_alignment_v5.py tests/test_alignment_charts_v5.py
```

The package was verified against the v4 test suite; 31 tests passed. Plotly chart functions were checked without launching the interactive Streamlit server, so please verify the visual layout locally.

**Privacy:** Keep traffic, forecasts, and reports on your local computer; this ZIP contains no company data.
