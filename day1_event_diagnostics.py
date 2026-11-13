"""Day-1 hourly forecast diagnostics with optional, auditable event annotations.

Use against the WashCast demand_ablation_* ``all_predictions.csv`` file.
This script NEVER retrains models or removes examples from the official scores.

Example:
    python day1_event_diagnostics.py \
       --predictions demand_ablation_aug_sep_all_origins/all_predictions.csv \
       --events events_review.csv \
       --model lgbm_stable_recent --reference weather_median \
       --output day1_diagnostics_aug_sep

Requires pandas/numpy. Events are diagnostic annotations, not model features.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

PREDICT_REQUIRED = {"model", "issued_at", "location_id", "local_date", "hour", "lead_day", "carwashes", "prediction"}
EVENT_REQUIRED = {"event_id", "start_date", "end_date", "location_id", "event_type", "status"}
KEY = ["issued_at", "location_id", "local_date", "hour", "lead_day"]


def metric(g: pd.DataFrame) -> pd.Series:
    actual = g.carwashes.to_numpy(float)
    err = g.error.to_numpy(float)
    ae = np.abs(err)
    total = actual.sum()
    return pd.Series({
        "hours": int(len(g)),
        "actual_avg": actual.mean(),
        "prediction_avg": g.prediction.mean(),
        "mae": ae.mean(),
        "median_ae": np.median(ae),
        "p90_ae": np.percentile(ae, 90),
        "signed_bias": err.mean(),
        "bias_pct_of_avg": 100 * err.sum() / total if total > 0 else np.nan,
        "wape_percent": 100 * ae.sum() / total if total > 0 else np.nan,
        "fraction_underpredicted": float((err < 0).mean()),
        "within_5_washes_pct": float((ae <= 5).mean() * 100),
    })


def metrics_by(d: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if d.empty:
        return pd.DataFrame(columns=keys + list(metric_template().keys()))
    return d.groupby(keys, sort=True, dropna=False).apply(metric, include_groups=False).reset_index()


def metric_template() -> dict:
    return {"hours": None, "actual_avg": None, "prediction_avg": None, "mae": None,
            "median_ae": None, "p90_ae": None, "signed_bias": None,
            "bias_pct_of_avg": None, "wape_percent": None,
            "fraction_underpredicted": None, "within_5_washes_pct": None}


def load_predictions(path: str, model: str, reference: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path)
    missing = PREDICT_REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    frame = frame.loc[(frame.lead_day == 1) & frame.model.isin([model, reference])].copy()
    if not {model, reference}.issubset(set(frame.model)):
        raise ValueError(f"Both model={model} and reference={reference} required in predictions")
    frame["local_date"] = pd.to_datetime(frame["local_date"]).dt.normalize()
    frame["hour"] = pd.to_numeric(frame["hour"], errors="raise").astype(int)
    frame["carwashes"] = pd.to_numeric(frame["carwashes"], errors="raise")
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="raise")
    if (~np.isfinite(frame[["carwashes", "prediction"]].to_numpy())).any():
        raise ValueError("Nonfinite prediction/actual values")
    if (frame.carwashes < 0).any() or (frame.prediction < 0).any():
        raise ValueError("Negative actual or prediction values")
    if frame.duplicated(["model", *KEY]).any():
        raise ValueError("Duplicate model forecast for same issuance/store/target hour")
    if "inferred_zero" not in frame:
        frame["inferred_zero"] = False
    frame["inferred_zero"] = frame.inferred_zero.astype(str).str.lower().isin(["true", "1", "yes"])
    frame["error"] = frame.prediction - frame.carwashes
    frame["absolute_error"] = frame.error.abs()
    a = frame[frame.model == model].copy()
    b = frame[frame.model == reference].copy()
    check = a[KEY + ["carwashes"]].merge(b[KEY + ["carwashes"]], on=KEY,
                                           how="outer", suffixes=("_model", "_reference"),
                                           indicator=True, validate="one_to_one")
    if (check._merge != "both").any() or not np.allclose(check.carwashes_model, check.carwashes_reference):
        raise ValueError("Models/reference are not paired on identical actual store-hours")
    return a, b


def read_events(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=list(EVENT_REQUIRED) + ["start_hour", "end_hour", "notes", "source_url"])
    events = pd.read_csv(path, dtype=str).fillna("")
    missing = EVENT_REQUIRED - set(events.columns)
    if missing:
        raise ValueError(f"Missing event ledger columns: {sorted(missing)}")
    events = events[events.event_id.str.strip().ne("")].copy()
    if events.event_id.duplicated().any():
        raise ValueError("Event IDs must be unique")
    for optional in ["start_hour", "end_hour", "notes", "source_url", "known_before_issuance"]:
        if optional not in events:
            events[optional] = ""
    # Candidate rows with unknown dates are valid in the ledger but do not annotate forecasts.
    events["start_dt"] = pd.to_datetime(events.start_date, errors="coerce").dt.normalize()
    events["end_dt"] = pd.to_datetime(events.end_date, errors="coerce").dt.normalize()
    if ((events.start_dt.notna() & events.end_dt.notna()) & (events.end_dt < events.start_dt)).any():
        raise ValueError("An event ends before it starts")
    return events


def annotate_events(a: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    a = a.copy()
    a["event_ids"] = ""
    a["event_types"] = ""
    a["event_statuses"] = ""
    a["confirmed_exceptional_event"] = False
    if events.empty:
        return a
    # Exclude unlocated events from mapping; do not invent AL code -> public store map.
    # Annotate at hour resolution without duplicating prediction rows when events overlap.
    ids = [[] for _ in range(len(a))]
    types = [[] for _ in range(len(a))]
    states = [[] for _ in range(len(a))]
    confirmed = np.zeros(len(a), dtype=bool)
    dates = a.local_date.to_numpy()
    hours = a.hour.to_numpy()
    loc = a.location_id.astype(str).to_numpy()
    for ev in events.itertuples(index=False):
        if pd.isna(ev.start_dt) or pd.isna(ev.end_dt):
            continue
        if str(ev.location_id).upper() in {"UNMAPPED", "UNKNOWN", ""}:
            continue
        match = (dates >= np.datetime64(ev.start_dt)) & (dates <= np.datetime64(ev.end_dt))
        if str(ev.location_id).upper() != "ALL":
            match &= loc == str(ev.location_id)
        if str(ev.start_hour).strip():
            match &= hours >= int(float(ev.start_hour))
        if str(ev.end_hour).strip():
            match &= hours <= int(float(ev.end_hour))
        for i in np.flatnonzero(match):
            ids[i].append(str(ev.event_id))
            types[i].append(str(ev.event_type))
            states[i].append(str(ev.status))
            if str(ev.status).strip().lower() == "confirmed" and str(ev.event_type).strip().lower() in {
                "closure", "grand_opening", "promotion", "free_wash", "maintenance", "reporting_outage",
                "special_event", "holiday_promotion", "known_capacity_reduction",
            }:
                confirmed[i] = True
    a["event_ids"] = [";".join(x) for x in ids]
    a["event_types"] = [";".join(x) for x in types]
    a["event_statuses"] = [";".join(x) for x in states]
    a["confirmed_exceptional_event"] = confirmed
    return a



def attach_historical_weather(a: pd.DataFrame, weather_csv: str, rain_threshold_mm: float = 0.1) -> pd.DataFrame:
    """Retrospective comparison only: historical weather is never a forecast input.

    NOTE historical_weather.csv may be reconstructed/reanalysis rather than gauge-observed rain.
    Do not assume these labels establish the exact conditions at a wash location.
    """
    if 'timestamp' not in a.columns or 'precipitation' not in a.columns:
        raise ValueError('Predictions need timestamp and forecast precipitation for weather comparison')
    w = pd.read_csv(weather_csv, usecols=['location_id', 'timestamp', 'precipitation'])
    w['timestamp'] = pd.to_datetime(w.timestamp, utc=True, errors='raise')
    if w.duplicated(['location_id', 'timestamp']).any():
        raise ValueError('Historical weather has duplicate location/time rows')
    w = w.rename(columns={'precipitation': 'historical_precipitation'})
    a = a.copy()
    a['timestamp'] = pd.to_datetime(a.timestamp, utc=True, errors='raise')
    a = a.merge(w, on=['location_id','timestamp'], how='left', validate='many_to_one')
    a['forecast_rain_flag'] = a.precipitation > rain_threshold_mm
    a['historical_rain_flag'] = a.historical_precipitation > rain_threshold_mm
    present = a.historical_precipitation.notna()
    a['weather_agreement'] = np.select(
        [present & a.forecast_rain_flag & a.historical_rain_flag,
         present & a.forecast_rain_flag & ~a.historical_rain_flag,
         present & ~a.forecast_rain_flag & a.historical_rain_flag,
         present & ~a.forecast_rain_flag & ~a.historical_rain_flag],
        ['both_rain','forecast_only','historical_only','both_dry'], default='historical_unavailable'
    )
    return a


def make_days(a: pd.DataFrame) -> pd.DataFrame:
    keys = ["issued_at", "location_id", "local_date"]
    days = a.groupby(keys, sort=True).agg(
        hours=("hour", "size"), actual_total=("carwashes", "sum"),
        predicted_total=("prediction", "sum"),
        hourly_mae=("absolute_error", "mean"),
        hourly_signed_bias=("error", "mean"),
        inferred_zero_hours=("inferred_zero", "sum"),
        rainy_forecast_hours=("precipitation", lambda v: int((v.fillna(0) > 0).sum())) if "precipitation" in a else ("hour", "size"),
        confirmed_event_hours=("confirmed_exceptional_event", "sum"),
        labeled_event_hours=("event_ids", lambda s: int(s.ne("").sum())),
        event_ids=("event_ids", lambda s: ";".join(sorted({i for v in s for i in v.split(";") if i}))),
    ).reset_index()
    if "precipitation" not in a:
        days["rainy_forecast_hours"] = 0
    days["expected_hours"] = np.where(days.local_date.dt.dayofweek == 6, 10, 11)
    days["complete_day"] = days.hours == days.expected_hours
    days["daily_error"] = days.predicted_total - days.actual_total
    days["daily_absolute_error"] = days.daily_error.abs()
    days["daily_pct_error"] = np.where(days.actual_total > 0, 100 * days.daily_error / days.actual_total, np.nan)
    days["suspect_zero_day"] = days.inferred_zero_hours >= 3
    return days


def make_network(days: pd.DataFrame) -> pd.DataFrame:
    eligible = days[days.complete_day].copy()
    eligible["overpredicted"] = eligible.daily_error > 0
    eligible["underpredicted"] = eligible.daily_error < 0
    net = eligible.groupby("local_date", sort=True).agg(
        stores=("location_id", "nunique"),
        total_actual=("actual_total", "sum"), total_predicted=("predicted_total", "sum"),
        stores_overpredicted=("overpredicted", "sum"),
        stores_underpredicted=("underpredicted", "sum"),
        inferred_zero_hours=("inferred_zero_hours", "sum"),
        rainy_forecast_hours=("rainy_forecast_hours", "sum"),
        stores_with_event=("labeled_event_hours", lambda x: int((x > 0).sum()))
    ).reset_index()
    net["network_bias_washes"] = net.total_predicted - net.total_actual
    net["network_bias_pct"] = np.where(net.total_actual > 0, 100 * net.network_bias_washes / net.total_actual, np.nan)
    net["fraction_stores_same_direction"] = net[["stores_overpredicted", "stores_underpredicted"]].max(axis=1) / net.stores
    return net.sort_values("local_date")


def make_persistent_bias(days: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Daily average hourly bias avoids conflating 10h Sundays with 11h weekdays.
    ordered = days.sort_values(["location_id", "local_date"]).copy()
    # Rolling windows use calendar days rather than rows (important when a store-day is absent).
    rolling_rows = []
    for loc, part in ordered.groupby("location_id", sort=True):
        part = part.set_index("local_date").sort_index()
        for w in [7, 14, 28]:
            r = part["hourly_signed_bias"].rolling(f"{w}D", min_periods=1)
            part[f"rolling_{w}d_hourly_bias"] = r.mean()
            part[f"rolling_{w}d_days"] = r.count()
        rolling_rows.append(part.reset_index())
    roll = pd.concat(rolling_rows, ignore_index=True) if rolling_rows else ordered
    roll["weekly_period"] = roll.local_date.dt.to_period("W-SUN").astype(str)
    weekly = roll.groupby(["location_id", "weekly_period"]).agg(
        evaluated_days=("local_date", "nunique"),
        avg_hourly_bias=("hourly_signed_bias", "mean"),
        avg_hourly_mae=("hourly_mae", "mean"),
        avg_actual_daily_total=("actual_total", "mean"),
    ).reset_index()
    summary = []
    for loc, part in roll.groupby("location_id", sort=True):
        signs = np.sign(part.daily_error.to_numpy())
        # Longest same-direction streak by *consecutive evaluated calendar dates*.
        longest = 0
        streak = 0
        last_sign = 0
        last_date = None
        for date, sign in zip(part.local_date, signs):
            if sign and sign == last_sign and last_date is not None and (date - last_date).days == 1:
                streak += 1
            elif sign:
                streak = 1
            else:
                streak = 0
            longest = max(longest, streak)
            last_date, last_sign = date, sign
        summary.append({
            "location_id": loc, "days": len(part),
            "daily_overprediction_fraction": float((signs > 0).mean()),
            "daily_underprediction_fraction": float((signs < 0).mean()),
            "avg_hourly_bias": part.hourly_signed_bias.mean(),
            "avg_hourly_mae": part.hourly_mae.mean(),
            "daily_bias_pct_of_volume": 100 * part.daily_error.sum() / part.actual_total.sum() if part.actual_total.sum() > 0 else np.nan,
            "longest_consecutive_same_sign_days": longest,
            "latest_14d_bias": float(part.iloc[-1].rolling_14d_hourly_bias),
            "latest_28d_bias": float(part.iloc[-1].rolling_28d_hourly_bias),
        })
    return roll, weekly, pd.DataFrame(summary)


def make_paired_rain(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    rename = b[KEY + ["prediction", "absolute_error", "error"]].rename(
        columns={"prediction": "reference_prediction", "absolute_error": "reference_absolute_error", "error": "reference_error"})
    paired = a.merge(rename, on=KEY, how="inner", validate="one_to_one")
    paired["gain_vs_reference"] = paired.reference_absolute_error - paired.absolute_error
    cols = ["issued_at", "location_id", "local_date", "weather_regime"]
    if "weather_regime" not in paired:
        paired["weather_regime"] = "unknown"
    events = paired.groupby(cols, sort=True).agg(
        hours=("hour", "size"),
        model_mae=("absolute_error", "mean"), reference_mae=("reference_absolute_error", "mean"),
        gain_vs_reference=("gain_vs_reference", "mean"),
        model_bias=("error", "mean"), reference_bias=("reference_error", "mean"),
        inferred_zero_hours=("inferred_zero", "sum"),
        annotated_event_hours=("event_ids", lambda s: int(s.ne("").sum())),
    ).reset_index()
    return events


def metric_scope(a: pd.DataFrame, label: str) -> dict:
    if a.empty:
        return {"scope": label, "hours": 0}
    values = metric(a)
    return {"scope": label, **values.to_dict()}


def run(predictions: str, output: str, model: str = "lgbm_stable_recent",
        reference: str = "weather_median", events_path: str | None = None,
        historical_weather_csv: str | None = None, rain_threshold_mm: float = 0.1) -> dict:
    if model == reference:
        raise ValueError("Choose different model and reference")
    a, b = load_predictions(predictions, model, reference)
    events = read_events(events_path)
    a = annotate_events(a, events)
    if historical_weather_csv:
        a = attach_historical_weather(a, historical_weather_csv, rain_threshold_mm)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    days = make_days(a)
    a.to_csv(output_dir / "day1_hourly_residuals.csv", index=False)
    days.to_csv(output_dir / "day1_store_days.csv", index=False)
    metrics_by(a, ["location_id", "hour"]).to_csv(output_dir / "store_hour_patterns.csv", index=False)
    metrics_by(a, ["location_id", "weather_regime"] if "weather_regime" in a else ["location_id"]).to_csv(
        output_dir / "store_weather_patterns.csv", index=False)
    roll, weekly, persistent = make_persistent_bias(days)
    roll.to_csv(output_dir / "store_rolling_bias.csv", index=False)
    weekly.to_csv(output_dir / "store_weekly_bias.csv", index=False)
    persistent.to_csv(output_dir / "store_persistent_bias.csv", index=False)
    network = make_network(days)
    network.to_csv(output_dir / "network_synchronous_days.csv", index=False)
    paired_rain = make_paired_rain(a, b)
    paired_rain.to_csv(output_dir / "store_day_weather_vs_reference.csv", index=False)
    if historical_weather_csv:
        metrics_by(a, ['weather_agreement']).to_csv(output_dir / 'weather_forecast_vs_historical_regimes.csv', index=False)
        metrics_by(a, ['location_id', 'weather_agreement']).to_csv(output_dir / 'weather_forecast_vs_historical_by_store.csv', index=False)
    # Sensitivity is descriptive: an event only excluded from this optional report when
    # explicitly recorded and marked confirmed. NEVER change the headline/official evaluation.
    bad_keys = days.loc[days.suspect_zero_day, ["issued_at", "location_id", "local_date"]]
    day_bad = a[KEY[:1] + ["location_id", "local_date"]].merge(
        bad_keys.assign(_bad=True), on=["issued_at", "location_id", "local_date"], how="left")
    suspect = day_bad._bad.eq(True).to_numpy(dtype=bool)
    confirmed = a.confirmed_exceptional_event.to_numpy(bool)
    sensitivity = pd.DataFrame([
        metric_scope(a, "all_official_hours"),
        metric_scope(a.loc[~confirmed], "exclude_confirmed_annotated_event_hours_diagnostic_only"),
        metric_scope(a.loc[~suspect], "exclude_days_with_3plus_inferred_zeros_diagnostic_only"),
        metric_scope(a.loc[~(suspect | confirmed)], "exclude_both_diagnostic_only"),
    ])
    sensitivity.to_csv(output_dir / "event_data_quality_sensitivity.csv", index=False)
    summary = {
        "model": model, "reference": reference, "lead_day": 1,
        "evaluated_hours": int(len(a)), "evaluated_store_days": int(len(days)),
        "complete_store_days": int(days.complete_day.sum()),
        "confirmed_event_annotated_hours": int(a.confirmed_exceptional_event.sum()),
        "days_with_3plus_inferred_zeros": int(days.suspect_zero_day.sum()),
        "events_without_mapped_locations_or_dates_are_not_automatically_joined": True,
        "historical_weather_coverage_fraction": float(a.historical_precipitation.notna().mean()) if historical_weather_csv else None,
        "historical_weather_is_retrospective_and_may_be_reanalysis_not_ground_truth": True,
        "note": "Event annotations and exclusions are retrospective diagnostics only, not independent proof of causation.",
    }
    (output_dir / "diagnostic_info.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Day-1 audit: {len(a)} hours, {len(days)} store-days ({days.complete_day.sum()} complete).")
    print("\nPersistent store diagnostics (largest absolute daily percent bias first):")
    print(persistent.reindex(persistent.daily_bias_pct_of_volume.abs().sort_values(ascending=False).index).head(8).round(2).to_string(index=False))
    print("\nLargest shared network misses:")
    print(network.reindex(network.network_bias_pct.abs().sort_values(ascending=False).index).head(8).round(2).to_string(index=False))
    print(f"\nSaved to {output_dir}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True)
    p.add_argument("--events", default=None, help="Optional CSV of reviewed events; never silently excludes training data")
    p.add_argument("--model", default="lgbm_stable_recent")
    p.add_argument("--reference", default="weather_median")
    p.add_argument("--output", required=True)
    p.add_argument("--historical-weather-csv", help="Optional post-event weather comparison; never as-issued predictors")
    p.add_argument("--rain-threshold-mm", type=float, default=0.1, help="Rain flag threshold for forecast vs historical mm/hour")
    args = p.parse_args()
    run(args.predictions, args.output, args.model, args.reference, args.events,
        args.historical_weather_csv, args.rain_threshold_mm)


if __name__ == "__main__":
    main()
