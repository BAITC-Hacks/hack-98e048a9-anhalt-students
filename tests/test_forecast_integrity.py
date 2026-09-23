"""Regression tests for provenance, chronology, weather units and physical bounds."""

from datetime import date
import json

import numpy as np
import pandas as pd
import pytest
import requests

from backend.agent import data_loader, weather_tool
from backend.agent.data_loader import get_or_create_historical_data
from backend.agent.model import WindForecastingModel
from backend.agent.physics import WindTurbinePhysics
from backend.agent.weather_tool import WeatherAgentTool


def history_frame(count=48):
    frame = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=count, freq="h"),
        "wind_speed_10m": np.linspace(3, 8, count),
        "wind_speed_100m": np.linspace(5, 12, count),
        "temperature_2m": np.zeros(count),
        "normalized_power": np.linspace(0, 1, count),
    })
    frame.attrs.update(data_source="SYNTHETIC_DEMO", is_synthetic=True, is_verified=False)
    return frame


class RecordingRegressor:
    def __init__(self):
        self.targets = []

    def fit(self, features, target):
        self.targets.append(target.copy())
        return self

    def predict(self, features):
        return np.full(len(features), 0.5)


class Response:
    def __init__(self, payload=None, error=False):
        self.payload = payload
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise requests.HTTPError("HTTP 503")

    def json(self):
        return self.payload


@pytest.fixture
def remote_weather(monkeypatch, tmp_path):
    monkeypatch.setenv("DEMO_MOCK_MODE", "false")
    monkeypatch.setattr(weather_tool, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(WeatherAgentTool, "_today", staticmethod(lambda: date(2026, 9, 23)))
    return WeatherAgentTool()


def test_training_sorts_history_and_excludes_forecast_origin():
    frame = history_frame().iloc[::-1].copy()
    model = WindForecastingModel()
    recorder = RecordingRegressor()
    model.model = recorder
    metrics = model.train(frame, cutoff="2026-01-02")
    assert metrics["fitted_samples"] == 24
    assert metrics["train_end"] == "2026-01-01T23:00:00"
    assert metrics["validation_train_end"] < metrics["validation_start"]
    assert np.array_equal(recorder.targets[-1], np.linspace(0, 1, 48)[:24])
    assert metrics["data_source"] == "SYNTHETIC_DEMO"
    assert metrics["is_synthetic"] is True


@pytest.mark.parametrize("corruption", ["duplicate", "out_of_range", "nan"])
def test_training_rejects_invalid_targets_and_duplicate_time(corruption):
    frame = history_frame()
    if corruption == "duplicate":
        frame.loc[1, "time"] = frame.loc[0, "time"]
    else:
        frame.loc[0, "normalized_power"] = 2 if corruption == "out_of_range" else np.nan
    with pytest.raises(ValueError):
        WindForecastingModel().train(frame)


def test_model_enforces_same_cutin_as_physics():
    model = WindForecastingModel()
    model.model = RecordingRegressor()
    model.is_trained = True
    weather = history_frame(4)
    weather["wind_speed_100m"] = [2.8, 3.1, 24.9, 25]
    assert np.array_equal(model.predict(weather), [0, 0.5, 0.5, 0])


def test_air_density_preserves_low_pressure_instead_of_clipping():
    density = WindTurbinePhysics.calculate_air_density(np.array([30.0]), np.array([750.0]))
    assert density[0] == pytest.approx(75000 / (287.058 * 303.15))
    assert density[0] < 1.0


def test_bundled_history_is_explicitly_synthetic():
    frame = get_or_create_historical_data()
    assert frame.attrs["data_source"] == "SYNTHETIC_DEMO"
    assert frame.attrs["is_synthetic"] is True
    assert frame.attrs["is_verified"] is False


def test_user_history_precedes_bundled_example(monkeypatch, tmp_path):
    supplied = history_frame(10)
    supplied.to_csv(tmp_path / "shelek_historical.csv", index=False)
    (tmp_path / "historical_wind.csv").write_text("invalid bundled example")
    monkeypatch.setattr(data_loader, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(data_loader, "HISTORICAL_CSV", str(tmp_path / "historical_wind.csv"))
    loaded = get_or_create_historical_data()
    assert len(loaded) == 10
    assert loaded.attrs["data_source"] == "USER_SUPPLIED_UNVERIFIED"
    assert loaded.attrs["is_verified"] is False


def test_each_wind_column_converts_its_own_units(remote_weather, monkeypatch):
    payload = remote_weather._generate_fallback_weather("2026-02-01", "2026-02-01")
    original = payload["hourly"]["wind_speed_10m"].copy()
    payload["hourly"]["wind_speed_10m"] = (np.array(original) * 3.6).tolist()
    payload["hourly_units"]["wind_speed_10m"] = "km/h"
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response(payload))
    result = remote_weather.fetch_forecast(43.6, 78.5, "2026-02-01", "2026-02-01")
    assert result["wind_speed_10m"].to_numpy() == pytest.approx(original)
    assert result["wind_speed_100m"].tolist() == payload["hourly"]["wind_speed_100m"]
    assert result.attrs["forecast_issue_time_verified"] is False


def test_forecast_hub_wind_interpolation_uses_documented_levels(remote_weather):
    payload = remote_weather._generate_fallback_weather("2026-02-01", "2026-02-01")
    for column in ("wind_speed_100m", "wind_direction_100m"):
        payload["hourly"].pop(column)
    payload["hourly"].update(wind_speed_80m=[8] * 24, wind_speed_120m=[12] * 24,
                            wind_direction_80m=[350] * 24, wind_direction_120m=[10] * 24)
    payload["hourly_units"].update(wind_speed_80m="m/s", wind_speed_120m="m/s")
    frame = remote_weather._decode_weather(payload, "2026-02-01", "2026-02-01")
    assert frame["wind_speed_100m"].eq(10).all()
    assert np.allclose(frame["wind_direction_100m"] % 360, 0, atol=1e-12) or np.allclose(frame["wind_direction_100m"], 360)
    assert "80m_120m" in frame.attrs["hub_wind_method"]


def test_archive_fallback_has_reanalysis_provenance(remote_weather, monkeypatch):
    payload = remote_weather._generate_fallback_weather("2026-02-01", "2026-02-01")
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        return Response(payload, error=len(calls) == 1)
    monkeypatch.setattr(requests, "get", get)
    result = remote_weather.fetch_forecast(43.6, 78.5, "2026-02-01", "2026-02-01")
    assert calls == [remote_weather.forecast_api_url, remote_weather.archive_api_url]
    assert result["data_source"].eq("OPEN_METEO_REANALYSIS").all()
    assert result.attrs["weather_kind"] == "retrospective_reanalysis"
    assert result.attrs["forecast_issue_time_verified"] is False


def test_future_uses_live_endpoint(remote_weather, monkeypatch):
    payload = remote_weather._generate_fallback_weather("2026-09-24", "2026-09-24")
    calls = []
    def get(url, **kwargs):
        calls.append((url, kwargs["params"]))
        return Response(payload)
    monkeypatch.setattr(requests, "get", get)
    result = remote_weather.fetch_forecast(43.6, 78.5, "2026-09-24", "2026-09-24")
    assert calls[0][0] == remote_weather.live_api_url
    assert "wind_speed_80m" in calls[0][1]["hourly"]
    assert result.attrs["weather_kind"] == "live_forecast"


def test_incomplete_remote_weather_falls_back_without_real_cache(remote_weather, monkeypatch, tmp_path):
    payload = remote_weather._generate_fallback_weather("2026-02-01", "2026-02-01")
    payload["hourly"]["wind_speed_100m"][3] = None
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response(payload))
    result = remote_weather.fetch_forecast(43.6, 78.5, "2026-02-01", "2026-02-01")
    assert len(result) == 24
    assert result.attrs["weather_kind"] == "synthetic_demo"
    assert "ValueError" in result.attrs["fallback_reason"]
    assert list(tmp_path.iterdir()) == []


def test_mock_dates_are_consistent_across_overlapping_horizons(remote_weather, monkeypatch, tmp_path):
    monkeypatch.setenv("DEMO_MOCK_MODE", "true")
    def unexpected(*args, **kwargs):
        pytest.fail("Mock mode must not access the network")
    monkeypatch.setattr(requests, "get", unexpected)
    long = remote_weather.fetch_forecast(43.6, 78.5, "2026-02-01", "2026-02-02")
    short = remote_weather.fetch_forecast(43.6, 78.5, "2026-02-02", "2026-02-02")
    pd.testing.assert_frame_equal(long.iloc[24:].reset_index(drop=True), short, check_flags=False)
    assert list(tmp_path.iterdir()) == []


def test_cache_is_reusable_and_force_refresh_bypasses_it(remote_weather, monkeypatch):
    payload = remote_weather._generate_fallback_weather("2026-02-01", "2026-02-01")
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        return Response(payload)
    monkeypatch.setattr(requests, "get", get)
    remote_weather.fetch_forecast(43.6, 78.5, "2026-02-01", "2026-02-01")
    cached = remote_weather.fetch_forecast(43.6, 78.5, "2026-02-01", "2026-02-01")
    assert cached.attrs["cache_hit"] is True
    assert len(calls) == 1
    refreshed = remote_weather.fetch_forecast(43.6, 78.5, "2026-02-01", "2026-02-01", force_refresh=True)
    assert refreshed.attrs["cache_hit"] is False
    assert len(calls) == 2


def test_legacy_unverified_cache_is_not_reused(remote_weather, monkeypatch, tmp_path):
    cache_file = tmp_path / "43.600000_78.500000_2026-02-01_2026-02-01.json"
    cache_file.write_text(json.dumps([{"time": "2026-02-01", "data_source": "OPEN_METEO_HISTORICAL_FORECAST"}]))
    def unavailable(*args, **kwargs):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(requests, "get", unavailable)
    frame = remote_weather.fetch_forecast(43.6, 78.5, "2026-02-01", "2026-02-01")
    assert frame.attrs["weather_kind"] == "synthetic_demo"


@pytest.mark.parametrize("start,end", [("2026-02-30", "2026-03-01"), ("2026-02-03", "2026-02-01")])
def test_invalid_weather_dates_are_rejected(remote_weather, start, end):
    with pytest.raises(ValueError):
        remote_weather.fetch_forecast(43.6, 78.5, start, end)


def test_pipeline_transactional_retraining_preserves_state_on_failure(tmp_path):
    from backend.agent.pipeline import SamrukWindAgentPipeline

    pipe = SamrukWindAgentPipeline()
    orig_history_len = len(pipe.history)
    orig_model = pipe.model

    target_csv = tmp_path / "candidate_scada.csv"
    target_csv.write_text("dummy_content_to_protect", encoding="utf-8")

    # Candidate with fewer than 10 rows prior to cutoff
    dates = pd.date_range("2026-02-15", periods=5, freq="h")
    bad_df = pd.DataFrame({
        "time": dates,
        "normalized_power": [0.5] * 5,
        "temperature_2m": [5.0] * 5,
        "wind_speed_10m": [8.0] * 5,
        "wind_speed_100m": [10.0] * 5,
    })
    bad_df.attrs.update(data_source="USER_SUPPLIED_UNVERIFIED", is_verified=False)

    with pytest.raises(ValueError, match="At least 10 historical records"):
        pipe.update_history_and_retrain(bad_df, target_path=str(target_csv))

    # Verify disk file was NOT overwritten with bad_df
    assert target_csv.read_text(encoding="utf-8") == "dummy_content_to_protect"
    # Verify in-memory history and model untouched
    assert len(pipe.history) == orig_history_len
    assert pipe.model is orig_model


def test_pipeline_transactional_retraining_success(tmp_path):
    from backend.agent.pipeline import SamrukWindAgentPipeline

    pipe = SamrukWindAgentPipeline()
    target_csv = tmp_path / "valid_scada.csv"

    dates = pd.date_range("2026-01-01", periods=24, freq="h")
    good_df = pd.DataFrame({
        "time": dates,
        "normalized_power": [0.65] * 24,
        "temperature_2m": [-3.0] * 24,
        "wind_speed_10m": [7.5] * 24,
        "wind_speed_100m": [9.5] * 24,
    })
    good_df.attrs.update(data_source="USER_SUPPLIED_UNVERIFIED", is_verified=False)

    metadata = pipe.update_history_and_retrain(good_df, target_path=str(target_csv))

    assert metadata["retrained"] is True
    assert metadata["data_source"] == "USER_SUPPLIED_UNVERIFIED"
    assert target_csv.is_file()
    assert len(pipe.history) == 24
    assert pipe.history.attrs["data_source"] == "USER_SUPPLIED_UNVERIFIED"


def test_real_pipeline_storm_scenario_enforces_cutout_zero_generation():
    from backend.agent.pipeline import SamrukWindAgentPipeline

    pipe = SamrukWindAgentPipeline()
    res = pipe.run_forecast_cycle("2026-02-14", 24, "farm", storm_scenario=True)
    assert res["status"] == "SUCCESS"
    assert res["audit"]["agent_status"] == "CRITICAL_SHUTDOWN"
    assert res["audit"]["alerts"]["storm_cutout_detected"] is True
    assert res["audit"]["total_generation_mwh"] == 0.0
    assert all(row["predicted_power"] == 0.0 for row in res["timeline"])
    assert "ДАУЫЛ" in res["audit"]["dispatcher_brief_kz"].upper()
    assert "ШТОРМ" in res["audit"]["dispatcher_brief_ru"].upper()



