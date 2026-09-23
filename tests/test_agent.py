import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from backend.agent.weather_tool import WeatherAgentTool, TURBINES
from backend.agent.physics import WindTurbinePhysics
from backend.agent.model import WindForecastingModel
from backend.agent.pipeline import SamrukWindAgentPipeline
from backend.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def offline_test_weather(monkeypatch):
    monkeypatch.setenv("DEMO_MOCK_MODE", "true")

def test_weather_tool_fetch_and_units():
    tool = WeatherAgentTool()
    coords = TURBINES["turbine_1"]
    df = tool.fetch_forecast(coords["lat"], coords["lon"], "2026-02-01", "2026-02-02")
    assert not df.empty
    assert "temperature_2m" in df.columns
    assert "wind_speed_100m" in df.columns
    assert "wind_speed_10m" in df.columns
    assert len(df) >= 24

    # CRITICAL CHECK: wind speeds must be in m/s (not km/h)
    # Typical realistic wind speeds in Shelek are 2 to 24 m/s (if in km/h it would be 20 to 90 km/h)
    mean_speed = df["wind_speed_100m"].mean()
    assert 1.0 <= mean_speed <= 25.0, f"Mean wind speed {mean_speed} looks like it is not in m/s!"

def test_physics_power_curve():
    # Test cut-in (< 3.0 m/s)
    zero_wind = np.array([1.0, 2.0])
    p_zero = WindTurbinePhysics.theoretical_power_curve(zero_wind)
    assert np.all(p_zero == 0.0)

    # Test rated power (12.5 - 24.9 m/s)
    rated_wind = np.array([13.0, 15.0])
    p_rated = WindTurbinePhysics.theoretical_power_curve(rated_wind)
    assert np.all(p_rated == 1.0)

    # Test storm cut-out (>= 25 m/s)
    storm_wind = np.array([25.0, 28.0, 32.0])
    p_storm = WindTurbinePhysics.theoretical_power_curve(storm_wind)
    assert np.all(p_storm == 0.0)

    # Test air density boost at cold winter temperatures (-15C vs +15C)
    rho_winter = WindTurbinePhysics.calculate_air_density(np.array([-15.0]), np.array([1013.25]))
    rho_standard = WindTurbinePhysics.calculate_air_density(np.array([15.0]), np.array([1013.25]))
    assert float(rho_winter[0]) > float(rho_standard[0]), "Cold air density must exceed standard air density!"

def test_turbine_capacity_scaling():
    pipe = SamrukWindAgentPipeline()
    # Single turbine (2.5 MW)
    res_single = pipe.run_forecast_cycle(target_date="2026-02-14", horizon_hours=24, turbine_id="turbine_1")
    assert res_single["metadata"]["rated_capacity_mw"] == 2.5
    for row in res_single["timeline"]:
        assert row["predicted_mwh"] <= 2.501, "Single turbine generation exceeded 2.5 MW rated capacity!"

    # Farm (5.0 MW)
    res_farm = pipe.run_forecast_cycle(target_date="2026-02-14", horizon_hours=24, turbine_id="farm")
    assert res_farm["metadata"]["rated_capacity_mw"] == 5.0
    assert res_farm["audit"]["total_generation_mwh"] > res_single["audit"]["total_generation_mwh"]

def test_end_to_end_agentic_pipeline_and_benchmark():
    pipe = SamrukWindAgentPipeline()
    res = pipe.run_forecast_cycle(target_date="2026-02-14", horizon_hours=24, turbine_id="turbine_1")
    assert res["status"] == "SUCCESS"
    assert "audit" in res
    assert "benchmark" in res
    assert "timeline" in res
    assert len(res["timeline"]) == 24
    
    # Verify bugfix: theoretical_power is calculated and non-zero
    theoretical_vals = [r["theoretical_power"] for r in res["timeline"]]
    assert any(v > 0.0 for v in theoretical_vals), "theoretical_power should be properly calculated and non-zero!"

    # Verify benchmark metrics
    assert "skill_score_index" in res["benchmark"]
    assert "mae_model_vs_physics" in res["benchmark"]
    assert res["audit"]["dispatcher_brief_ru"]
    assert res["audit"]["dispatcher_brief_kz"]

def test_fastapi_endpoints():
    # Test GET /api/turbines
    r_turbines = client.get("/api/turbines")
    assert r_turbines.status_code == 200
    assert "turbines" in r_turbines.json()

    # Test GET /api/forecast
    r_fc = client.get("/api/forecast?target_date=2026-02-14&horizon_hours=24&turbine_id=turbine_2")
    assert r_fc.status_code == 200
    assert r_fc.json()["metadata"]["turbine_id"] == "turbine_2"

    # Test GET /api/export/csv with turbine_id
    r_csv = client.get("/api/export/csv?target_date=2026-02-14&horizon_hours=24&turbine_id=turbine_2")
    assert r_csv.status_code == 200
    assert "wind_forecast_turbine_2_2026-02-14.csv" in r_csv.headers.get("content-disposition", "")
    assert "predicted_mwh" in r_csv.text
