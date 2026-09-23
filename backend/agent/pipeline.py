"""Bounded, auditable orchestration. Accuracy requires observed generation."""
from datetime import datetime, timedelta, timezone
import hashlib
import os
from threading import RLock
from uuid import uuid4
import numpy as np
import pandas as pd
from .weather_tool import WeatherAgentTool, TURBINES
from .physics import WindTurbinePhysics
from .model import WindForecastingModel
from .reasoner import DispatcherAgentReasoner
from .data_loader import get_or_create_historical_data


class SamrukWindAgentPipeline:
    def __init__(self):
        self.weather_tool = WeatherAgentTool()
        self.model = WindForecastingModel()
        self._lock = RLock()
        self._runs = {}
        self.history = get_or_create_historical_data()
        self.training_metadata = self.model.train(self.history, cutoff="2026-02-01")
        self._training_cutoff = pd.Timestamp("2026-02-01")

    @staticmethod
    def _trace(trace, step, action, reason, **details):
        trace.append(dict(step=step, action=action, reason=reason, details=details))

    @staticmethod
    def _validate_weather(frame, start, hours):
        required = ["time", "wind_speed_10m", "temperature_2m", "surface_pressure"]
        if any(col not in frame for col in required):
            raise ValueError("Weather response is missing required columns")
        frame = frame.copy()
        frame["time"] = pd.to_datetime(frame["time"], errors="raise")
        expected = pd.date_range(start, periods=hours, freq="h")
        frame = frame.loc[frame["time"].isin(expected)].sort_values("time").reset_index(drop=True)
        if not frame["time"].equals(pd.Series(expected)):
            raise ValueError("Weather must contain each requested hour exactly once")
        numeric = required[1:] + (["wind_speed_100m"] if "wind_speed_100m" in frame else [])
        if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
            raise ValueError("Weather contains non-finite values")
        if (frame["wind_speed_10m"] < 0).any() or (frame["surface_pressure"] <= 0).any():
            raise ValueError("Weather contains invalid wind or pressure")
        if "wind_speed_100m" in frame and (frame["wind_speed_100m"] < 0).any():
            raise ValueError("Weather contains negative hub-height wind")
        if (frame["temperature_2m"] <= -273.15).any():
            raise ValueError("Weather temperature must exceed absolute zero")
        return frame

    def run_forecast_cycle(self, target_date="2026-02-14", horizon_hours=48,
                           turbine_id="turbine_1", refresh_weather=False, storm_scenario=False):
        if turbine_id in ("all", "both", "cluster"):
            turbine_id = "farm"
        if turbine_id not in TURBINES:
            raise ValueError("Unknown turbine_id")
        start = datetime.strptime(target_date, "%Y-%m-%d")
        if not isinstance(horizon_hours, int) or not 1 <= horizon_hours <= 168:
            raise ValueError("horizon_hours must be between 1 and 168")
        with self._lock:
            cutoff = min(pd.Timestamp(start), pd.Timestamp("2026-02-01"))
            if cutoff != getattr(self, "_training_cutoff", pd.Timestamp("2026-02-01")):
                self.training_metadata = self.model.train(self.history, cutoff=cutoff)
                self._training_cutoff = cutoff
            if storm_scenario:
                if turbine_id == "farm":
                    return self._combine_farm([self._run_single(target_date, horizon_hours, tid, refresh_weather, storm_scenario=True)
                                               for tid in ("turbine_1", "turbine_2")])
                return self._run_single(target_date, horizon_hours, turbine_id, refresh_weather, storm_scenario=True)
            if turbine_id == "farm":
                return self._combine_farm([self._run_single(target_date, horizon_hours, tid, refresh_weather)
                                           for tid in ("turbine_1", "turbine_2")])
            return self._run_single(target_date, horizon_hours, turbine_id, refresh_weather)

    def _run_single(self, target_date, hours, turbine_id, refresh_weather, storm_scenario=False):
        turbine = TURBINES[turbine_id]
        capacity = float(turbine["rated_mw"])
        start = datetime.strptime(target_date, "%Y-%m-%d")
        end_date = (start + timedelta(hours=hours - 1)).strftime("%Y-%m-%d")
        trace = []
        self._trace(trace, "plan", "fetch_validate_predict_audit", "Produce a complete hourly forecast", hours=hours)
        fetch_kwargs = {"storm_scenario": True} if storm_scenario else {}
        for attempt in range(2):
            try:
                raw = self.weather_tool.fetch_forecast(turbine["lat"], turbine["lon"], target_date,
                                                       end_date, force_refresh=refresh_weather or attempt > 0,
                                                       **fetch_kwargs)
                frame = self._validate_weather(raw, start, hours)
                self._trace(trace, "retrieve", "weather_accepted", "Complete finite hourly weather grid", attempt=attempt + 1)
                break
            except (ValueError, KeyError, TypeError) as exc:
                self._trace(trace, "validate", "refresh_weather", str(exc), attempt=attempt + 1)
                if attempt == 1:
                    raise ValueError("Weather validation failed after one refresh") from exc
        fingerprint = hashlib.sha256((frame.to_json(date_format="iso") + str(storm_scenario)).encode()).hexdigest()
        key = (target_date, hours, turbine_id, storm_scenario)
        previous = self._runs.get(key)
        changed = previous is not None and previous["fingerprint"] != fingerprint
        revision = previous["revision"] + int(changed) if previous else 1
        self._trace(trace, "update", "recalculate" if changed else "calculate",
                    "Weather inputs changed" if changed else "Initial or unchanged weather inputs",
                    input_changed=changed, revision=revision, fingerprint=fingerprint)
        enriched = WindTurbinePhysics.enrich_features(frame)
        self._trace(trace, "prepare", "enrich_physics", "Air density, hub-height wind and generic power curve")
        model_kind = "LightGBM with physical features"
        for attempt in range(2):
            try:
                predicted = (np.asarray(self.model.predict(enriched), dtype=float) if attempt == 0
                             else enriched["theoretical_power"].to_numpy(dtype=float).copy())
                if predicted.shape != (hours,) or not np.isfinite(predicted).all():
                    raise ValueError("Model produced incomplete or non-finite predictions")
                if ((predicted < 0) | (predicted > 1)).any():
                    raise ValueError("Model predictions exceed normalized power bounds")
                break
            except Exception as exc:
                if attempt == 1:
                    raise ValueError("Both ML and physics inference failed") from exc
                model_kind = "Generic physical curve (ML fallback)"
                self._trace(trace, "analyze", "use_physics_fallback", type(exc).__name__ + ": " + str(exc))
        wind = enriched["wind_speed_100m"].to_numpy()
        stopped = (wind >= WindTurbinePhysics.V_CUT_OUT) | (wind < WindTurbinePhysics.V_CUT_IN)
        revised = int(np.count_nonzero(predicted[stopped]))
        predicted[stopped] = 0.0
        enriched["predicted_power"] = predicted
        self._trace(trace, "forecast", "apply_operating_bounds", "Enforce cut-in and cut-out limits", revised_hours=revised)
        audit = DispatcherAgentReasoner.audit_forecast(enriched, target_date, capacity_mw=capacity)
        self._trace(trace, "analyze", "dispatcher_audit", "Rule-based weather and ramp checks",
                    events=len(audit["events"]), status=audit["agent_status"])
        if revised:
            self._trace(trace, "revise", "rechecked_after_correction", "Audit uses the corrected generation schedule")
        source = str(frame["data_source"].iloc[0]) if "data_source" in frame else "UNKNOWN"
        training_source = self.training_metadata.get("data_source", "UNVERIFIED")
        physics = enriched["theoretical_power"].to_numpy()
        timeline = []
        for idx, row in enriched.iterrows():
            timeline.append({
                "hour_index": int(idx), "timestamp": pd.Timestamp(row["time"]).isoformat(),
                "wind_speed_10m": round(float(row["wind_speed_10m"]), 2),
                "wind_speed_100m": round(float(row["wind_speed_100m"]), 2),
                "temperature_2m": round(float(row["temperature_2m"]), 1),
                "air_density_kg_m3": round(float(row["air_density"]), 4),
                "theoretical_power": round(float(physics[idx]), 4),
                "predicted_power": round(float(predicted[idx]), 4),
                "predicted_mwh": round(float(predicted[idx]) * capacity, 6),
                "physics_mwh": round(float(physics[idx]) * capacity, 6),
                "persistence_mwh": None, "data_source": source})
        self._trace(trace, "finalize", "publish_forecast", "No observed test generation: accuracy and savings are not evaluated")
        self._runs[key] = {"fingerprint": fingerprint, "revision": revision}
        if len(self._runs) > 512:
            del self._runs[next(iter(self._runs))]
        return {
            "status": "SUCCESS",
            "metadata": {
                "run_id": str(uuid4()), "turbine_id": turbine_id, "turbine_name": turbine["name"],
                "rated_capacity_mw": capacity, "coordinates": f"{turbine['lat']:.6f}, {turbine['lon']:.6f}",
                "target_date": target_date, "horizon_hours": hours, "timezone": "Asia/Almaty",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "scenario_issue_time": (start - timedelta(hours=1)).isoformat(),
                "forecast_issue_time_verified": False,
                "mode": "RETROSPECTIVE_SCENARIO" if start.date() < datetime.now().date() else "FORECAST_SCENARIO",
                "input_fingerprint": fingerprint, "input_changed": changed, "revision": revision,
                "training_data_source": training_source, "training": self.training_metadata,
                "weather": {field: frame.attrs.get(field) for field in (
                    "weather_kind", "hub_wind_method", "fallback_reason", "cache_hit", "retrieved_at")},
                "update_policy": "Recheck inputs on each request; refresh_weather bypasses weather cache"},
            "audit": audit,
            "benchmark": {
                "model_type": model_kind, "baseline_physical": "Generic density-adjusted turbine curve (not IEC certified)",
                "baseline_naive": "Unavailable: verified turbine observations required",
                "mae_model_vs_physics": round(float(np.mean(np.abs(predicted - physics))), 4),
                "mae_persistence": None, "skill_score_index": None,
                "evaluation_status": "UNVERIFIED_NO_ACTUALS", "data_provenance": source,
                "training_data_source": training_source},
            "agent_trace": trace, "timeline": timeline}

    def _combine_farm(self, parts):
        """Sum independent turbine forecasts instead of using midpoint weather."""
        first, second = parts
        timeline = []
        for a, b in zip(first["timeline"], second["timeline"]):
            row = dict(a)
            for field in ("wind_speed_10m", "wind_speed_100m", "temperature_2m", "air_density_kg_m3", "theoretical_power", "predicted_power"):
                row[field] = (a[field] + b[field]) / 2.0
            for field in ("predicted_mwh", "physics_mwh"):
                row[field] = round(a[field] + b[field], 6)
            row["data_source"] = " + ".join(sorted({a["data_source"], b["data_source"]}))
            timeline.append(row)
        frame = pd.DataFrame(timeline).rename(columns={"timestamp": "time", "air_density_kg_m3": "air_density"})
        audit = DispatcherAgentReasoner.audit_forecast(frame, first["metadata"]["target_date"], capacity_mw=5.0)
        audit["events"] = [dict(event, turbine_id=part["metadata"]["turbine_id"]) for part in parts for event in part["audit"]["events"]]
        for key in ("storm_hours", "icing_hours", "high_ramp_hours"):
            audit["alerts"][key] = sorted({h for part in parts for h in part["audit"]["alerts"][key]})
        audit["alerts"]["storm_cutout_detected"] = bool(audit["alerts"]["storm_hours"])
        audit["alerts"]["icing_risk_detected"] = bool(audit["alerts"]["icing_hours"])
        audit["agent_status"] = ("CRITICAL_SHUTDOWN" if audit["alerts"]["storm_hours"] else
                                  "ADVISORY_ATTENTION" if audit["events"] else "VERIFIED_STABLE")
        for lang in ("ru", "kz"):
            audit[f"dispatcher_brief_{lang}"] = "\n\n".join(part["metadata"]["turbine_name"] + "\n" + part["audit"][f"dispatcher_brief_{lang}"] for part in parts)
        meta = dict(first["metadata"], run_id=str(uuid4()), turbine_id="farm", turbine_name=TURBINES["farm"]["name"], rated_capacity_mw=5.0,
                    weather={part["metadata"]["turbine_id"]: part["metadata"]["weather"] for part in parts},
                    coordinates="; ".join(part["metadata"]["coordinates"] for part in parts),
                    input_changed=any(part["metadata"]["input_changed"] for part in parts),
                    revision=max(part["metadata"]["revision"] for part in parts),
                    input_fingerprint=hashlib.sha256("".join(part["metadata"]["input_fingerprint"] for part in parts).encode()).hexdigest())
        benchmark = dict(first["benchmark"], data_provenance=" + ".join(sorted({part["benchmark"]["data_provenance"] for part in parts})),
                         model_type=" + ".join(sorted({part["benchmark"]["model_type"] for part in parts})),
                         mae_model_vs_physics=round(float(np.mean(np.abs(frame["predicted_power"] - frame["theoretical_power"]))), 4))
        return {"status": "SUCCESS", "metadata": meta, "audit": audit, "benchmark": benchmark, "timeline": timeline,
                "agent_trace": [dict(event, turbine_id=part["metadata"]["turbine_id"]) for part in parts for event in part["agent_trace"]],
                "turbine_forecasts": parts}

    def run_full_february_simulation(self, turbine_id="farm"):
        days = {}
        for day in range(1, 29):
            date = f"2026-02-{day:02d}"
            result = self.run_forecast_cycle(date, 24, turbine_id)
            days[date] = {"generation_mwh": sum(row["predicted_mwh"] for row in result["timeline"]),
                          "physics_mwh": sum(row["physics_mwh"] for row in result["timeline"]),
                          "capacity_factor": result["audit"]["capacity_factor_pct"],
                          "status": result["audit"]["agent_status"], "events_count": len(result["audit"]["events"]),
                          "data_source": result["benchmark"]["data_provenance"]}
        total = sum(day["generation_mwh"] for day in days.values())
        storm_days = [d for d, val in days.items() if val["status"] == "CRITICAL_SHUTDOWN"]
        advisory_days = [d for d, val in days.items() if val["status"] == "ADVISORY_ATTENTION"]
        return {"month": "February 2026", "turbine_id": turbine_id, "total_days": 28,
                "total_monthly_mwh": round(total, 2), "total_physics_mwh": round(sum(day["physics_mwh"] for day in days.values()), 2),
                "mean_daily_mwh": round(total / 28, 2),
                "storm_shutdown_days": storm_days,
                "advisory_days": advisory_days,
                "daily_breakdown": days,
                "evaluation_status": "RETROSPECTIVE_SCENARIO_NOT_ACCURACY_BACKTEST", "forecast_issue_time_verified": False}

    def update_history_and_retrain(self, candidate_df: pd.DataFrame, target_path: str = None) -> dict:
        """
        Transactional update of historical data and forecasting model:
        1. Test-train a fresh candidate model on candidate_df before modifying any state.
        2. If training succeeds, safely write candidate_df to target_path (if provided).
        3. Under self._lock, update self.model, self.history, self.training_metadata, and clear self._runs.
        4. If anything fails (e.g. ValueError due to insufficient records), previous state and files are completely untouched.
        """
        candidate_model = WindForecastingModel()
        metadata = candidate_model.train(candidate_df, cutoff="2026-02-01")

        if target_path:
            tmp_target = f"{target_path}.tmp"
            candidate_df.to_csv(tmp_target, index=False)
            os.replace(tmp_target, target_path)

        with self._lock:
            self.model = candidate_model
            self.history = candidate_df.copy()
            self.training_metadata = metadata
            self._training_cutoff = pd.Timestamp("2026-02-01")
            self._runs.clear()

        return dict(metadata, retrained=True)

    def reload_data_and_retrain(self):
        """Thread-safe reload of historical SCADA dataset with candidate rollback on failure."""
        candidate_history = get_or_create_historical_data()
        return self.update_history_and_retrain(candidate_history)

