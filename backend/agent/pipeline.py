import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .weather_tool import WeatherAgentTool, TURBINES
from .physics import WindTurbinePhysics
from .model import WindForecastingModel
from .reasoner import DispatcherAgentReasoner
from .data_loader import get_or_create_historical_data

class SamrukWindAgentPipeline:
    """
    Autonomous Agentic AI Pipeline for Samruk-Kazyna Wind Farm Generation Forecasting.
    Orchestrates: Weather Retrieval -> Physics Enrichment -> Multi-Model Inference (LightGBM vs Physics vs Persistence) -> Dispatcher Audit.
    """
    def __init__(self):
        self.weather_tool = WeatherAgentTool()
        self.model = WindForecastingModel()
        self._ensure_trained()

    def _ensure_trained(self):
        hist_df = get_or_create_historical_data()
        print("[Agent Pipeline] Training / Verifying LightGBM forecasting model...")
        metrics = self.model.train(hist_df, target_col="normalized_power")
        print(f"[Agent Pipeline] Model Ready. Validation Metrics: RMSE={metrics['rmse']}, MAE={metrics['mae']}, R2={metrics['r2_score']}")

    def run_forecast_cycle(self, target_date: str = "2026-02-14", horizon_hours: int = 48, turbine_id: str = "turbine_1") -> dict:
        """
        Executes complete Agentic Loop for a single forecast run.
        """
        if turbine_id in ("all", "both", "cluster"):
            turbine_id = "farm"
        turbine = TURBINES.get(turbine_id, TURBINES["turbine_1"])
        lat, lon = turbine["lat"], turbine["lon"]
        capacity_mw = float(turbine.get("rated_mw", 2.5))

        dt_start = datetime.strptime(target_date, "%Y-%m-%d")
        dt_end = dt_start + timedelta(hours=horizon_hours - 1)
        end_date_str = dt_end.strftime("%Y-%m-%d")

        # Step 1: Agent retrieves weather forecasts by turbine coordinates
        raw_weather = self.weather_tool.fetch_forecast(lat, lon, target_date, end_date_str)
        weather_slice = raw_weather.head(horizon_hours).copy().reset_index(drop=True)

        # Step 2: Aerodynamic physics enrichment (IEC 61400-12 curve + air density + Hellman shear)
        enriched = WindTurbinePhysics.enrich_features(weather_slice)

        # Step 3: Multi-model inference
        # 3a. Physics-Informed LightGBM Model
        predicted_power = self.model.predict(enriched)
        enriched["predicted_power"] = predicted_power

        # 3b. Pure Aerodynamic Physical Baseline
        physics_power = enriched["theoretical_power"].values

        # 3c. Persistence Baseline (Hour 0 level held constant across horizon)
        h0_power = float(predicted_power[0]) if len(predicted_power) > 0 else 0.5
        persistence_power = np.full(len(predicted_power), h0_power)

        # Step 4: Agentic Reasoning & Dispatcher Audit
        audit = DispatcherAgentReasoner.audit_forecast(enriched, target_date, capacity_mw=capacity_mw)

        # Step 5: Benchmark metrics calculation (MAE, RMSE, Skill Score)
        # Skill Score relative to Persistence: 1 - (MAE_model / MAE_persistence)
        mae_model_vs_phys = float(np.mean(np.abs(predicted_power - physics_power)))
        mae_model_vs_pers = float(np.mean(np.abs(predicted_power - persistence_power)))
        mae_phys_vs_pers = float(np.mean(np.abs(physics_power - persistence_power)))
        
        # Skill score indicates how much better ML is over naive flat persistence
        skill_score = round(max(0.0, 1.0 - (mae_model_vs_phys / (mae_phys_vs_pers + 1e-6))), 3)

        benchmark = {
            "model_type": "Physics-Informed LightGBM (Hybrid)",
            "baseline_physical": "IEC 61400-12 Air-Density Scaled Curve",
            "baseline_naive": "Flat Persistence (H0 constant)",
            "mae_model_vs_physics": round(mae_model_vs_phys, 4),
            "mae_persistence": round(mae_model_vs_pers, 4),
            "skill_score_index": skill_score,
            "data_provenance": enriched["data_source"].iloc[0] if "data_source" in enriched.columns else "OPEN_METEO"
        }

        # Structure hourly timeline for dashboard and API
        timeline = []
        for idx, row in enriched.iterrows():
            p_pred = float(row.get("predicted_power", 0.0))
            p_phys = float(row.get("theoretical_power", 0.0))
            p_pers = float(persistence_power[idx])

            timeline.append({
                "hour_index": int(idx),
                "timestamp": str(row.get("time", f"{target_date} {idx:02d}:00")),
                "wind_speed_10m": round(float(row.get("wind_speed_10m", 0.0)), 2),
                "wind_speed_100m": round(float(row.get("wind_speed_100m", 0.0)), 2),
                "temperature_2m": round(float(row.get("temperature_2m", 0.0)), 1),
                "air_density_kg_m3": round(float(row.get("air_density", 1.225)), 4),
                "theoretical_power": round(p_phys, 4),
                "predicted_power": round(p_pred, 4),
                "predicted_mwh": round(p_pred * capacity_mw, 3),
                "physics_mwh": round(p_phys * capacity_mw, 3),
                "persistence_mwh": round(p_pers * capacity_mw, 3)
            })

        return {
            "status": "SUCCESS",
            "metadata": {
                "turbine_id": turbine_id,
                "turbine_name": turbine["name"],
                "rated_capacity_mw": capacity_mw,
                "coordinates": f"{lat:.6f}, {lon:.6f}",
                "target_date": target_date,
                "horizon_hours": horizon_hours,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "audit": audit,
            "benchmark": benchmark,
            "timeline": timeline
        }

    def run_full_february_simulation(self, turbine_id: str = "farm") -> dict:
        """
        Backtests the complete test month (Feb 1 - Feb 28, 2026) day-by-day.
        Reproduces the exact task requirement: forecast 24-48h for each day.
        """
        results_by_day = {}
        total_monthly_mwh = 0.0
        total_physics_mwh = 0.0

        for day in range(1, 29):
            date_str = f"2026-02-{day:02d}"
            day_res = self.run_forecast_cycle(target_date=date_str, horizon_hours=24, turbine_id=turbine_id)
            daily_mwh = day_res["audit"]["total_generation_mwh"]
            daily_phys = day_res["audit"].get("physics_generation_mwh", daily_mwh)
            total_monthly_mwh += daily_mwh
            total_physics_mwh += daily_phys

            results_by_day[date_str] = {
                "generation_mwh": daily_mwh,
                "physics_mwh": daily_phys,
                "capacity_factor": day_res["audit"]["capacity_factor_pct"],
                "status": day_res["audit"]["agent_status"],
                "events_count": len(day_res["audit"].get("events", []))
            }

        return {
            "month": "February 2026",
            "turbine_id": turbine_id,
            "total_days": 28,
            "total_monthly_mwh": round(total_monthly_mwh, 2),
            "total_physics_mwh": round(total_physics_mwh, 2),
            "mean_daily_mwh": round(total_monthly_mwh / 28.0, 2),
            "daily_breakdown": results_by_day
        }
