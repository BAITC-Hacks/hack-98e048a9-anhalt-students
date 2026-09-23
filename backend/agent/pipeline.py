import os
import pandas as pd
from datetime import datetime, timedelta
from .weather_tool import WeatherAgentTool, TURBINES
from .physics import WindTurbinePhysics
from .model import WindForecastingModel
from .reasoner import DispatcherAgentReasoner
from .data_loader import get_or_create_historical_data

class SamrukWindAgentPipeline:
    """
    Autonomous Agentic AI Pipeline for Samruk-Kazyna Wind Farm Generation Forecasting.
    Orchestrates: Weather Retrieval -> Physics Enrichment -> ML Inference -> Dispatcher Audit.
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
        turbine = TURBINES.get(turbine_id, TURBINES["turbine_1"])
        lat, lon = turbine["lat"], turbine["lon"]

        dt_start = datetime.strptime(target_date, "%Y-%m-%d")
        dt_end = dt_start + timedelta(hours=horizon_hours - 1)
        end_date_str = dt_end.strftime("%Y-%m-%d")

        # Step 1: Agent retrieves weather forecasts by turbine coordinates
        raw_weather = self.weather_tool.fetch_forecast(lat, lon, target_date, end_date_str)
        # Take exactly horizon_hours
        weather_slice = raw_weather.head(horizon_hours).copy()

        # Step 2 & 3: Physics feature enrichment & ML model prediction
        predictions = self.model.predict(weather_slice)
        weather_slice["predicted_power"] = predictions

        # Step 4: Agentic Reasoning & Dispatcher Audit
        audit = DispatcherAgentReasoner.audit_forecast(weather_slice, target_date)

        # Structure full response for dashboard and API
        timeline = []
        for idx, row in weather_slice.iterrows():
            timeline.append({
                "hour_index": int(idx),
                "timestamp": str(row.get("time", f"{target_date} {idx:02d}:00")),
                "wind_speed_10m": float(row.get("wind_speed_10m", 0)),
                "wind_speed_100m": float(row.get("wind_speed_100m", 0)),
                "temperature_2m": float(row.get("temperature_2m", 0)),
                "theoretical_power": float(row.get("theoretical_power", 0)),
                "predicted_power": float(row.get("predicted_power", 0)),
                "predicted_mwh": round(float(row.get("predicted_power", 0)) * 5.0, 3)
            })

        return {
            "status": "SUCCESS",
            "metadata": {
                "turbine": turbine["name"],
                "coordinates": f"{lat:.6f}, {lon:.6f}",
                "target_date": target_date,
                "horizon_hours": horizon_hours,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "audit": audit,
            "timeline": timeline
        }

    def run_full_february_simulation(self) -> dict:
        """
        Backtests the complete test month (Feb 1 - Feb 28, 2026) day-by-day.
        Reproduces the exact task requirement: forecast 24-48h for each day.
        """
        results_by_day = {}
        total_monthly_mwh = 0.0

        for day in range(1, 29):
            date_str = f"2026-02-{day:02d}"
            day_res = self.run_forecast_cycle(target_date=date_str, horizon_hours=24)
            daily_mwh = day_res["audit"]["total_generation_mwh"]
            total_monthly_mwh += daily_mwh
            results_by_day[date_str] = {
                "generation_mwh": daily_mwh,
                "capacity_factor": day_res["audit"]["capacity_factor_pct"],
                "alerts": day_res["audit"]["alerts"]
            }

        return {
            "month": "February 2026",
            "total_days": 28,
            "total_monthly_mwh": round(total_monthly_mwh, 2),
            "mean_daily_mwh": round(total_monthly_mwh / 28.0, 2),
            "daily_breakdown": results_by_day
        }
