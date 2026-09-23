from threading import RLock
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from backend.agent.pipeline import SamrukWindAgentPipeline
from backend.agent.weather_tool import TURBINES


def weather(speed=8.0):
    return pd.DataFrame({"time": pd.date_range("2026-02-14", periods=24, freq="h"),
                         "wind_speed_10m": speed, "wind_speed_100m": speed,
                         "temperature_2m": 10.0, "surface_pressure": 1000.0,
                         "data_source": "TEST_SYNTHETIC"})


@pytest.fixture
def pipe():
    instance = SamrukWindAgentPipeline.__new__(SamrukWindAgentPipeline)
    instance._lock = RLock()
    instance._runs = {}
    instance.training_metadata = {"train_end": "2026-01-31T23:00:00", "data_source": "TEST_SYNTHETIC"}
    instance.weather_tool = SimpleNamespace(fetch_forecast=lambda *a, **kw: weather())
    instance.model = SimpleNamespace(predict=lambda df: df["theoretical_power"].to_numpy().copy())
    return instance


def test_updated_inputs_trigger_recalculation_and_revision(pipe):
    first = pipe.run_forecast_cycle(horizon_hours=24)
    same = pipe.run_forecast_cycle(horizon_hours=24)
    pipe.weather_tool.fetch_forecast = lambda *a, **kw: weather(11)
    updated = pipe.run_forecast_cycle(horizon_hours=24, refresh_weather=True)
    assert same["metadata"]["revision"] == 1
    assert updated["metadata"]["revision"] == 2
    assert updated["metadata"]["input_changed"] is True
    assert updated["timeline"][0]["predicted_power"] != first["timeline"][0]["predicted_power"]
    assert any(step["action"] == "recalculate" for step in updated["agent_trace"])


def test_invalid_weather_refreshes_once_then_uses_valid_grid(pipe):
    calls = []
    def fetch(*args, force_refresh=False):
        calls.append(force_refresh)
        return weather().iloc[1:] if len(calls) == 1 else weather()
    pipe.weather_tool.fetch_forecast = fetch
    result = pipe.run_forecast_cycle(horizon_hours=24)
    assert calls == [False, True]
    assert len(result["timeline"]) == 24


def test_invalid_weather_is_bounded_and_rejected(pipe):
    calls = []
    def fetch(*args, **kwargs):
        calls.append(1)
        return weather().iloc[1:]
    pipe.weather_tool.fetch_forecast = fetch
    with pytest.raises(ValueError, match="after one refresh"):
        pipe.run_forecast_cycle(horizon_hours=24)
    assert len(calls) == 2


def test_model_nonfinite_output_selects_physics_fallback(pipe):
    pipe.model.predict = lambda df: np.full(len(df), np.nan)
    result = pipe.run_forecast_cycle(horizon_hours=24)
    assert "fallback" in result["benchmark"]["model_type"]
    assert all(np.isfinite(row["predicted_power"]) for row in result["timeline"])
    assert any(step["action"] == "use_physics_fallback" for step in result["agent_trace"])


def test_storm_output_corrected_before_audit(pipe):
    pipe.weather_tool.fetch_forecast = lambda *a, **kw: weather(26)
    pipe.model.predict = lambda df: np.ones(len(df))
    result = pipe.run_forecast_cycle(horizon_hours=24)
    assert result["audit"]["total_generation_mwh"] == 0
    assert result["audit"]["alerts"]["storm_cutout_detected"]
    assert any(step["action"] == "rechecked_after_correction" for step in result["agent_trace"])


def test_farm_sums_turbines_without_midpoint_shutdown(pipe):
    pipe.weather_tool.fetch_forecast = lambda lat, *a, **kw: weather(26 if lat == TURBINES["turbine_1"]["lat"] else 13)
    result = pipe.run_forecast_cycle(horizon_hours=24, turbine_id="farm")
    assert all(row["predicted_mwh"] == 2.5 for row in result["timeline"])
    assert result["audit"]["alerts"]["storm_cutout_detected"]
    assert result["audit"]["events"][0]["turbine_id"] == "turbine_1"
    assert set(result["metadata"]["weather"]) == {"turbine_1", "turbine_2"}


def test_unobserved_accuracy_and_savings_are_not_claimed(pipe):
    result = pipe.run_forecast_cycle(horizon_hours=24)
    assert result["benchmark"]["skill_score_index"] is None
    assert result["benchmark"]["mae_persistence"] is None
    assert result["audit"]["estimated_penalty_saved_kzt"] is None
    assert not result["metadata"]["forecast_issue_time_verified"]
    assert all(row["persistence_mwh"] is None for row in result["timeline"])


def test_training_cutoff_is_restored_after_earlier_request(pipe):
    cutoffs = []
    pipe.history = pd.DataFrame()
    def train(history, cutoff):
        cutoffs.append(pd.Timestamp(cutoff))
        return {"train_end": (pd.Timestamp(cutoff) - pd.Timedelta(hours=1)).isoformat(), "data_source": "TEST"}
    pipe.model.train = train
    pipe._run_single = lambda *args: {}
    pipe.run_forecast_cycle("2026-01-10", 24)
    pipe.run_forecast_cycle("2026-02-14", 24)
    assert cutoffs == [pd.Timestamp("2026-01-10"), pd.Timestamp("2026-02-01")]
