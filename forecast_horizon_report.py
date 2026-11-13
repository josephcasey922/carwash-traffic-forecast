"""Forecast reliability by lead day, store and complete operating day.

Example:
    python forecast_horizon_report.py \
      --predictions demand_ablation_aug_sep_all_origins/all_predictions.csv \
      --output forecast_horizon_aug_sep

Only requires pandas/numpy, already available in WashCast's environment.
This script does not change the forecasts or the training process.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = {"issued_at", "location_id", "local_date", "hour", "lead_day", "model", "carwashes", "prediction"}


def summarize_hourly(group: pd.DataFrame) -> pd.Series:
    y = group["carwashes"].to_numpy(dtype=float)
    p = group["prediction"].to_numpy(dtype=float)
    errors = p - y
    ae = np.abs(errors)
    total = y.sum()
    return pd.Series({
        "hours": len(group),
        "actual_mean_per_hour": y.mean(),
        "mae_washes_per_hour": ae.mean(),
        "median_ae_washes_per_hour": np.median(ae),
        "p90_ae_washes_per_hour": np.percentile(ae, 90),
        "wape_percent": 100 * ae.sum() / total if total > 0 else np.nan,
        "bias_washes_per_hour": errors.mean(),
        "within_2_washes_percent": 100 * np.mean(ae <= 2),
        "within_5_washes_percent": 100 * np.mean(ae <= 5),
        "within_10_washes_percent": 100 * np.mean(ae <= 10),
    })


def summarize_daily(group: pd.DataFrame) -> pd.Series:
    y = group["actual_total"].to_numpy(dtype=float)
    p = group["predicted_total"].to_numpy(dtype=float)
    err = p - y
    ae = np.abs(err)
    pos = y > 0
    ape = 100 * ae[pos] / y[pos]
    return pd.Series({
        "store_days": len(group),
        "actual_average_daily_washes": y.mean(),
        "mae_daily_total_washes": ae.mean(),
        "median_ae_daily_total_washes": np.median(ae),
        "p90_ae_daily_total_washes": np.percentile(ae, 90),
        "daily_total_wape_percent": 100 * ae.sum() / y.sum() if y.sum() > 0 else np.nan,
        "median_daily_absolute_pct_error": np.median(ape) if ape.size else np.nan,
        "p90_daily_absolute_pct_error": np.percentile(ape, 90) if ape.size else np.nan,
        "daily_total_bias_washes": err.mean(),
        "daily_total_bias_pct_of_actual": 100 * err.sum() / y.sum() if y.sum() > 0 else np.nan,
        "within_10pct_daily_total": 100 * np.mean(ape <= 10) if ape.size else np.nan,
        "within_20pct_daily_total": 100 * np.mean(ape <= 20) if ape.size else np.nan,
    })


def group_summary(data: pd.DataFrame, by: list[str], summarizer) -> pd.DataFrame:
    return data.groupby(by, dropna=False, sort=True).apply(summarizer, include_groups=False).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", default="", help="Optional comma-separated model filter")
    args = parser.parse_args()

    d = pd.read_csv(args.predictions)
    missing = REQUIRED.difference(d.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")
    if args.models:
        wanted = set(args.models.split(","))
        d = d[d.model.isin(wanted)].copy()
    d["lead_day"] = pd.to_numeric(d["lead_day"], errors="raise").astype(int)
    d["carwashes"] = pd.to_numeric(d["carwashes"], errors="raise")
    d["prediction"] = pd.to_numeric(d["prediction"], errors="raise")
    d["hour"] = pd.to_numeric(d["hour"], errors="raise").astype(int)
    if not np.isfinite(d[["prediction", "carwashes"]].to_numpy()).all():
        raise SystemExit("Nonfinite values in prediction or carwashes")
    if (d.carwashes < 0).any():
        raise SystemExit("Negative actual wash counts found")
    d["local_date"] = pd.to_datetime(d["local_date"], errors="raise").dt.date

    key = ["model", "issued_at", "location_id", "local_date", "lead_day", "hour"]
    duplicates = d.duplicated(key, keep=False)
    if duplicates.any():
        raise SystemExit(f"Found {int(duplicates.sum())} duplicate forecast-hour rows; inspect keys {key}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    hourly = group_summary(d, ["model", "lead_day"], summarize_hourly)
    hourly.to_csv(output / "hourly_by_lead_day.csv", index=False)
    group_summary(d[d.lead_day == 1], ["model", "location_id"], summarize_hourly).to_csv(output / "day1_hourly_by_store.csv", index=False)
    if "weather_regime" in d:
        group_summary(d[d.lead_day == 1], ["model", "weather_regime"], summarize_hourly).to_csv(output / "day1_hourly_by_weather.csv", index=False)

    daily_key = ["model", "issued_at", "location_id", "local_date", "lead_day"]
    days = d.groupby(daily_key, dropna=False, as_index=False).agg(
        actual_total=("carwashes", "sum"),
        predicted_total=("prediction", "sum"),
        observed_hours=("hour", "nunique"),
        earliest_hour=("hour", "min"),
        latest_hour=("hour", "max"),
    )
    dt = pd.to_datetime(days["local_date"])
    days["expected_hours"] = np.where(dt.dt.dayofweek == 6, 10, 11)
    days["expected_start_hour"] = np.where(dt.dt.dayofweek == 6, 9, 8)
    days["complete_day"] = (
        (days["observed_hours"] == days["expected_hours"])
        & (days["earliest_hour"] == days["expected_start_hour"])
        & (days["latest_hour"] == 18)
    )
    days["daily_total_error"] = days["predicted_total"] - days["actual_total"]
    days["absolute_daily_total_error"] = days["daily_total_error"].abs()
    days["absolute_daily_pct_error"] = np.where(
        days["actual_total"] > 0,
        100 * days["absolute_daily_total_error"] / days["actual_total"].replace(0, np.nan),
        np.nan,
    )
    days.to_csv(output / "individual_store_days.csv", index=False)
    complete = days[days.complete_day].copy()
    daily = group_summary(complete, ["model", "lead_day"], summarize_daily)
    daily.to_csv(output / "daily_totals_by_lead_day.csv", index=False)
    group_summary(complete[complete.lead_day == 1], ["model", "location_id"], summarize_daily).to_csv(output / "day1_daily_totals_by_store.csv", index=False)
    coverage = days.groupby(["model", "lead_day"], as_index=False).agg(
        all_store_days=("complete_day", "size"),
        complete_store_days=("complete_day", "sum"),
    )
    coverage["excluded_incomplete_store_days"] = coverage.all_store_days - coverage.complete_store_days
    coverage.to_csv(output / "day_coverage.csv", index=False)
    print("\nHOURLY FORECAST ERRORS BY LEAD DAY")
    print(hourly.round(2).to_string(index=False))
    print("\nFULL-DAY TOTAL ERRORS (only complete store-days)")
    print(daily.round(2).to_string(index=False))
    print("\nSTORE-DAY COVERAGE")
    print(coverage.to_string(index=False))
    print(f"\nReports saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
