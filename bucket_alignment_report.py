"""Score historical WashCast Day-1 wash forecasts against current traffic-to-staffing buckets.

This is an EVALUATION ONLY; it does not optimize or predict staffing.

    python bucket_alignment_report.py \
        --predictions demand_ablation_oct_nov_refreshed/all_predictions.csv \
        --output alignment_oct_nov

Outputs: overall.csv, by_month.csv, by_store.csv, by_store_month.csv,
         by_week.csv, bucket_confusion.csv, by_target_day.csv.
All scores are computed on the exact historical as-issued predictions.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = {"model", "issued_at", "location_id", "timestamp", "lead_day", "carwashes", "prediction"}
BUCKETS = ["0-17", "18-40", "41-60", "61+"]


def employee_bucket(washes: np.ndarray) -> np.ndarray:
    """Current operational rule: nearest integer washes, halves rounded up.

    For reproducibility this uses half-up rounding for nonnegative counts.
    It maps to employee counts 2, 3, 4, 5; NOT a model training target.
    """
    rounded = np.floor(np.maximum(np.asarray(washes, dtype=float), 0) + 0.5)
    return 2 + np.searchsorted([18, 41, 61], rounded, side="right")


def score(group: pd.DataFrame) -> pd.Series:
    y = group["carwashes"].to_numpy(dtype=float)
    p = group["prediction"].to_numpy(dtype=float)
    error = p - y
    rec_actual = employee_bucket(y)
    rec_pred = employee_bucket(p)
    diff = rec_pred - rec_actual
    return pd.Series({
        "hours": int(len(group)),
        "avg_actual_washes": float(y.mean()),
        "mae": float(np.abs(error).mean()),
        "median_ae": float(np.median(np.abs(error))),
        "wape_percent": float(100 * np.abs(error).sum() / y.sum()) if y.sum() > 0 else np.nan,
        "signed_bias": float(error.mean()),
        "bucket_agreement_percent": float(100 * (diff == 0).mean()),
        "bucket_under_percent": float(100 * (diff < 0).mean()),
        "bucket_over_percent": float(100 * (diff > 0).mean()),
        "recommended_employee_hours": int(rec_actual.sum()),
        "absolute_employee_hour_difference": int(np.abs(diff).sum()),
        "under_recommended_employee_hours": int(np.maximum(-diff, 0).sum()),
        "over_recommended_employee_hours": int(np.maximum(diff, 0).sum()),
        "forecast_labor_alignment_percent": float(100 * (1 - np.abs(diff).sum() / rec_actual.sum())),
        "inferred_zero_hours": int(group["inferred_zero"].sum()) if "inferred_zero" in group.columns else np.nan,
    })


def grouped(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return df.groupby(keys, observed=True, dropna=False, sort=True).apply(score, include_groups=False).reset_index()


def run(predictions: str, output: str, models: str = "") -> dict[str, pd.DataFrame]:
    df = pd.read_csv(predictions, dtype={"location_id": str})
    missing = REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f"Prediction file missing required columns: {sorted(missing)}")
    if models:
        names = [x.strip() for x in models.split(",") if x.strip()]
        not_found = set(names) - set(df.model.astype(str))
        if not_found:
            raise ValueError(f"Models not found in predictions: {sorted(not_found)}")
        df = df[df.model.isin(names)].copy()
    df["lead_day"] = pd.to_numeric(df.lead_day, errors="raise")
    df = df[df.lead_day == 1].copy()
    if df.empty:
        raise ValueError("No lead_day=1 observations available.")
    for col in ("carwashes", "prediction"):
        df[col] = pd.to_numeric(df[col], errors="raise")
    if not np.isfinite(df[["carwashes", "prediction"]].to_numpy()).all():
        raise ValueError("Non-finite wash counts or forecasts.")
    if (df[["carwashes", "prediction"]] < 0).any().any():
        raise ValueError("Negative wash counts or forecasts.")
    key = ["model", "issued_at", "location_id", "timestamp", "lead_day"]
    if df.duplicated(key).any():
        raise ValueError("Repeated model/issuance/store/target-hour rows: do not merge overlapping runs.")

    # Fair comparisons require all selected models scored on the same issued target hours.
    counts = df.groupby("model").size()
    if counts.nunique() > 1:
        raise ValueError("Models have different numbers of forecast hours; compare on matched hours first.")
    key_no_model = ["issued_at", "location_id", "timestamp", "lead_day"]
    base = None
    for _, sub in df.groupby("model"):
        keys = set(map(tuple, sub[key_no_model].itertuples(index=False, name=None)))
        if base is None:
            base = keys
        elif keys != base:
            raise ValueError("Different forecast target hours among models; compare on matched hours first.")

    if "local_date" in df:
        target_date = pd.to_datetime(df["local_date"], errors="raise")
    else:
        target_date = pd.to_datetime(df["timestamp"], errors="raise", utc=True).dt.tz_convert("America/Chicago").dt.tz_localize(None)
    df["target_day"] = target_date.dt.strftime("%Y-%m-%d")
    df["month"] = target_date.dt.strftime("%Y-%m")
    df["week"] = (target_date - pd.to_timedelta(target_date.dt.weekday, unit="D")).dt.strftime("%Y-%m-%d")
    if "inferred_zero" in df:
        df["inferred_zero"] = df["inferred_zero"].astype(str).str.lower().isin(("true", "1", "yes"))

    result = {
        "overall": grouped(df, ["model"]),
        "by_month": grouped(df, ["model", "month"]),
        "by_store": grouped(df, ["model", "location_id"]),
        "by_store_month": grouped(df, ["model", "location_id", "month"]),
        "by_week": grouped(df, ["model", "week"]),
        "by_target_day": grouped(df, ["model", "target_day"]),
    }
    # Reference-versus-prediction employee-level confusion for inspecting threshold misses.
    df["actual_employee_bucket"] = employee_bucket(df.carwashes)
    df["predicted_employee_bucket"] = employee_bucket(df.prediction)
    result["bucket_confusion"] = (
        df.groupby(["model", "actual_employee_bucket", "predicted_employee_bucket"],
                   observed=True).size().rename("hours").reset_index()
    )
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in result.items():
        data.to_csv(folder / (name + ".csv"), index=False)
    cols = ["model", "hours", "mae", "wape_percent", "bucket_agreement_percent",
            "forecast_labor_alignment_percent", "under_recommended_employee_hours"]
    print("\nNext-day rollout alignment (forecast-derived; NOT actual staffing):")
    print(result["overall"][cols].round(3).to_string(index=False))
    print("\nMonthly:")
    print(result["by_month"][["model", "month", "hours", "mae", "bucket_agreement_percent",
                              "forecast_labor_alignment_percent"]].round(3).to_string(index=False))
    print(f"\nSaved seven CSV reports under {folder.resolve()}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="alignment_oct_nov")
    parser.add_argument("--models", default="", help="Comma-separated names (default: all models in file)")
    args = parser.parse_args()
    run(args.predictions, args.output, args.models)


if __name__ == "__main__":
    main()
