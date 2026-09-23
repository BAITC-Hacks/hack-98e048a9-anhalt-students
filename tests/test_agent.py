import pytest
import numpy as np
import pandas as pd
from backend.agent.weather_tool import WeatherAgentTool, TURBINES
from backend.agent.physics import WindTurbinePhysics
from backend.agent.model import WindForecastingModel
from backend.agent.pipeline import SamrukWindAgentPipeline

def test_weather_tool_fetch():
    tool = WeatherAgentTool()
    coords = TURBINES["turbine_1"]
    df = tool.fetch_forecast(coords["lat"], coords["lon"], "2026-02-01", "2026-02-02")
    assert not df.empty
    assert "temperature_2m" in df.columns
    assert "wind_speed_10m" in df.columns
    assert len(df) >= 24

def test_physics_power_curve():
    # Test cut-in
    zero_wind = np.array([1.0, 2.0])
    p_zero = WindTurbinePhysics.theoretical_power_curve(zero_wind)
    assert np.all(p_zero == 0.0)

    # Test rated power
    rated_wind = np.array([13.0, 15.0])
    p_rated = WindTurbinePhysics.theoretical_power_curve(rated_wind)
    assert np.all(p_rated == 1.0)

    # Test storm cut-out (>= 25 m/s)
    storm_wind = np.array([25.5, 30.0])
    p_storm = WindTurbinePhysics.theoretical_power_curve(storm_wind)
    assert np.all(p_storm == 0.0)

def test_end_to_end_agentic_pipeline():
    pipe = SamrukWindAgentPipeline()
    res = pipe.run_forecast_cycle(target_date="2026-02-14", horizon_hours=24)
    assert res["status"] == "SUCCESS"
    assert "audit" in res
    assert "timeline" in res
    assert len(res["timeline"]) == 24
    assert res["audit"]["total_generation_mwh"] >= 0.0
    assert "dispatcher_brief_ru" in res["audit"]
