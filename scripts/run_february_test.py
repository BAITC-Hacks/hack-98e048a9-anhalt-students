"""Generate February forecast windows and a provenance-aware summary.

Default: 28 origins x 24 hours x 2 turbines = 1,344 rows.
A 48-hour run has 2,688 rows; only lead hours 1..24 enter monthly totals.
This is a forecast replay, not an accuracy backtest without measured generation.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_OUTPUT = ROOT / "data" / "submission_forecast_february_2026.csv"


def build_submission(pipeline=None, horizon_hours: int = 24, start_date: str = "2026-02-01", days: int = 28):
    """Return a CSV frame and summary without writing any output files."""
    if horizon_hours not in (24, 48):
        raise ValueError("February submission horizon must be 24 or 48 hours")
    if pipeline is None:
        from backend.agent.pipeline import SamrukWindAgentPipeline

        pipeline = SamrukWindAgentPipeline()

    start_dt = pd.Timestamp(start_date)
    origins = [(start_dt + pd.Timedelta(days=d)).strftime("%Y-%m-%d") for d in range(days)]
    rows = []
    daily_summaries = []
    for origin in origins:
        for turbine_id in ("turbine_1", "turbine_2"):
            result = pipeline.run_forecast_cycle(
                target_date=origin, horizon_hours=horizon_hours, turbine_id=turbine_id
            )
            timeline = result["timeline"]
            if len(timeline) != horizon_hours:
                raise ValueError(f"Incomplete forecast for {turbine_id} on {origin}")
            metadata = result.get("metadata", {})
            benchmark = result.get("benchmark", {})
            for index, row in enumerate(timeline):
                rows.append({
                    "forecast_origin_date": origin,
                    "forecast_target_date": origin,
                    "scenario_issue_time": metadata.get("scenario_issue_time"),
                    "generated_at": metadata.get("generated_at"),
                    "lead_hour": index + 1,
                    "timestamp": row["timestamp"],
                    "timezone": metadata.get("timezone", "Asia/Almaty"),
                    "turbine_id": turbine_id,
                    "predicted_mw": row.get("predicted_mw", row["predicted_mwh"]),
                    "predicted_mwh": row["predicted_mwh"],
                    "interval_hours": 1,
                    "physics_baseline_mw": row.get("physics_mw", row["physics_mwh"]),
                    "persistence_baseline_mw": row.get("persistence_mw", row["persistence_mwh"]),
                    "wind_speed_100m_ms": row["wind_speed_100m"],
                    "temperature_2m_c": row["temperature_2m"],
                    "air_density_kg_m3": row["air_density_kg_m3"],
                    "weather_source": benchmark.get("data_provenance", "UNKNOWN"),
                    "training_data_source": benchmark.get("training_data_source", metadata.get("training_data_source", "UNKNOWN")),
                    "evaluation_status": benchmark.get("evaluation_status", "UNVERIFIED_NO_ACTUALS"),
                    "forecast_issue_time_verified": metadata.get("forecast_issue_time_verified", False),
                    "included_in_monthly_total": index < 24,
                })
            daily_summaries.append({
                "date": origin,
                "turbine_id": turbine_id,
                "forecast_generation_first_24h_mwh": round(sum(row["predicted_mwh"] for row in timeline[:24]), 3),
                "agent_status": result["audit"]["agent_status"],
            })

    frame = pd.DataFrame(rows)
    accounting_rows = frame.loc[frame["included_in_monthly_total"]]
    if accounting_rows.duplicated(["turbine_id", "timestamp"]).any():
        raise ValueError("Duplicate turbine hours in monthly accounting")
    total_mwh = float(accounting_rows["predicted_mwh"].sum())
    summary = {
        "month": "2026-02",
        "start_date": origins[0],
        "end_date": origins[-1],
        "forecast_origins": len(origins),
        "turbines": 2,
        "horizon_hours": horizon_hours,
        "rows": len(frame),
        "monthly_accounting_turbine_hours": len(accounting_rows),
        "monthly_accounting_calendar_hours": len(origins) * 24,
        "monthly_forecast_generation_mwh": round(total_mwh, 3),
        "forecast_capacity_factor_pct": round(total_mwh / (len(origins) * 24 * 5.0) * 100, 3),
        "weather_sources": sorted(frame["weather_source"].unique().tolist()),
        "training_data_sources": sorted(frame["training_data_source"].unique().tolist()),
        "evaluation_status": "UNVERIFIED_NO_ACTUALS",
        "skill_score_vs_observations": None,
        "verified_kegoc_savings_kzt": None,
        "forecast_issue_time_verified": bool(frame["forecast_issue_time_verified"].all()),
        "forecast_origin_date_definition": "Legacy column name: target window start date; scenario_issue_time is the nominal issue time.",
        "accounting_method": "Sum lead hours 1..24 for both turbines at each February origin; exclude overlapping later leads.",
        "limitations": [
            "Predicted generation, not measured generation or verified forecast accuracy.",
            "Archived weather does not establish a forecast available at the historical issue time.",
            "No SCADA actuals or verified imbalance prices: accuracy and financial savings are not evaluated.",
        ],
        "daily_breakdown": daily_summaries,
    }
    return frame, summary


def run_test(output_path=DEFAULT_OUTPUT, horizon_hours: int = 24, pipeline=None, start_date: str = "2026-02-01", days: int = 28):
    """Write forecast rows and a companion summary; return the summary for callers."""
    load_dotenv(ROOT / ".env", override=False)
    os.environ.setdefault("DEMO_MOCK_MODE", "true")
    frame, summary = build_submission(pipeline, horizon_hours, start_date, days)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Forecast replay completed: {len(frame):,} rows; {horizon_hours}h windows for two turbines ({len(summary['daily_breakdown']) // 2} origins).")
    print(f"CSV: {output_path}")
    print(f"Summary: {summary_path}")
    print(f"February forecast energy (non-overlapping): {summary['monthly_forecast_generation_mwh']:,.3f} MWh")
    print(f"Forecast capacity factor: {summary['forecast_capacity_factor_pct']:.3f}%")
    print(f"Weather sources: {', '.join(summary['weather_sources'])}")
    print(f"Training sources: {', '.join(summary['training_data_sources'])}")
    print("Accuracy Skill Score and KEGOC savings: NOT EVALUATED (no observed generation / settlement data).")
    return summary


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CSV output path (existing file will be replaced)")
    parser.add_argument("--horizon-hours", type=int, choices=(24, 48), default=24)
    parser.add_argument("--start-date", type=str, default="2026-02-01", help="Start origin date (e.g. 2026-01-31)")
    parser.add_argument("--days", type=int, default=28, help="Number of daily origins (default: 28)")
    parser.add_argument("--walkforward-29", action="store_true", help="Run 29 walk-forward 48h cycles (2026-01-31 to 2026-02-28 = 2,784 rows)")
    parser.add_argument("--mock", action="store_true", help="Force deterministic offline weather")
    args = parser.parse_args()
    if args.mock:
        os.environ["DEMO_MOCK_MODE"] = "true"
    start_date = "2026-01-31" if args.walkforward_29 else args.start_date
    days = 29 if args.walkforward_29 else args.days
    horizon = 48 if args.walkforward_29 else args.horizon_hours
    run_test(args.output, horizon, start_date=start_date, days=days)


if __name__ == "__main__":
    main()
