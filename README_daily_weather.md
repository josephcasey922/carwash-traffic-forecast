# WashCast: daily archived weather downloads with no traffic CSV

This utility extends the **Open-Meteo Single Runs** archive from a `location_id` / grid-coordinate CSV, without needing any car wash transaction data. Future forecasts can later be joined to actual hourly wash counts on `location_id` and UTC `timestamp`/target hour, with `issued_at` identifying the forecast snapshot.

## Before running

- Place `fetch_daily_archives.py` beside `asof.py` and `washcast.py` in the WashCast project folder.
- Place `washcast_weather_grid_mapping.csv` in that folder (columns `location_id,api_grid_latitude,api_grid_longitude`).
- **Keep** `weather_data/daily_partial/archived_forecasts.csv` and `historical_weather.csv` from your prior transfer. This avoids recollecting March 1–June 9, 2026.
- Keep `.weather_cache` in your project root too; it may save additional requests. Older cache entries keyed to the **original** unknown store coordinates may not match requests made using the recovered grid centers. The *combined CSV* is therefore the key preservation step.
- Install dependencies from the project's `requirements.txt` in your virtual environment.
- Check that you have permission to use Open-Meteo's API for your intended purpose. The free tier is not a commercial license.

## Dry run (makes zero API calls)

In PowerShell, from the project directory:

```powershell
.\.venv\Scripts\python.exe fetch_daily_archives.py `
  --locations washcast_weather_grid_mapping.csv `
  --start-date 2026-06-10 `
  --end-date 2026-10-20 `
  --output weather_data/daily_partial `
  --dry-run
```

## Download missing archived forecasts, conservatively

```powershell
.\.venv\Scripts\python.exe fetch_daily_archives.py `
  --locations washcast_weather_grid_mapping.csv `
  --start-date 2026-06-10 `
  --end-date 2026-10-20 `
  --output weather_data/daily_partial `
  --max-new-requests 100 `
  --daily-request-cap 250
```

The script stops when it reaches 100 *new HTTP requests in this run* or 250 locally logged attempted requests that day, whichever comes first. Rerun it later to continue. Successful response bodies are stored in `.weather_cache`. The archive CSV is updated on stopping, including after interruption or API error.

**The cap is a local safety limit, not an Open-Meteo quota guarantee.** It doesn't know about calls made elsewhere or the provider's fractional/weighted API call accounting. This script leaves a large margin below 10,000; if Open-Meteo returns HTTP 429, it stops. It does not try to bypass any rate limits.

All data is saved in `weather_data/daily_partial/archived_forecasts.csv`. The original historical series in `weather_data/daily_partial/historical_weather.csv` remains untouched. Based on the ZIP reviewed, that historical series already covers Feb 25–Oct 21, 2026, for the mapped locations. **Beyond Oct 20 forecast issuances**, update historical weather for the later dates before training as-of forecasts (rainfall lag features require pre-issuance history).

## Notes on coordinates and compatibility

The mapping records **API-returned weather-grid coordinates**, not verified carwash GPS positions. These reproduce the **location-to-grid association** we recovered, but a fresh API query at a grid center could differ slightly from the original query due to the service's grid selection or elevation adjustment. Preserve the original archived CSV, and don't claim perfect identity between previously downloaded and newly requested weather until checked.

Use the **same `location_id` values** in future traffic and weather datasets. The future traffic CSV may carry different precise site coordinates; that does not affect joining weather by `location_id` and hourly UTC target time, though geographical model features should be reviewed for consistency.

This utility downloads only **historical issued forecasts**, never live forecasts; it intentionally does not fit any models or connect to proprietary traffic data.
