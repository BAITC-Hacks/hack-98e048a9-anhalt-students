"""HTTP boundaries and submission accounting, without weather or model side effects."""

from datetime import datetime, timedelta
import io
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import main
from scripts.run_february_test import build_submission, run_test


class StubPipeline:
    def __init__(self):
        self.calls = []

    def run_forecast_cycle(self, target_date, horizon_hours, turbine_id, refresh_weather=False):
        self.calls.append((target_date, horizon_hours, turbine_id, refresh_weather))
        power = {"turbine_1": 1.0, "turbine_2": 2.0, "farm": 3.0}[turbine_id]
        start = datetime.fromisoformat(target_date)
        timeline = [{
            "hour_index": index,
            "timestamp": (start + timedelta(hours=index)).isoformat(),
            "predicted_mwh": power,
            "physics_mwh": power,
            "persistence_mwh": power,
            "wind_speed_100m": 8.0,
            "temperature_2m": -5.0,
            "air_density_kg_m3": 1.2,
        } for index in range(horizon_hours)]
        return {
            "status": "SUCCESS",
            "metadata": {
                "turbine_id": turbine_id, "timezone": "Asia/Almaty", "forecast_issue_time_verified": False,
                "scenario_issue_time": (start - timedelta(hours=1)).isoformat(),
            },
            "timeline": timeline,
            "benchmark": {
                "skill_score_index": None,
                "data_provenance": "MOCK_MODE_SYNTHESIS",
                "training_data_source": "SYNTHETIC_DEMO",
                "evaluation_status": "UNVERIFIED_NO_ACTUALS",
            },
            "audit": {"agent_status": "NORMAL", "estimated_penalty_saved_kzt": None},
        }

    def run_full_february_simulation(self, turbine_id):
        return {"turbine_id": turbine_id}

    def reload_data_and_retrain(self):
        return {"retrained": True, "data_source": "USER_SUPPLIED_UNVERIFIED"}

    def update_history_and_retrain(self, candidate_df, target_path=None):
        return {"retrained": True, "rows": len(candidate_df), "data_source": "USER_SUPPLIED_UNVERIFIED"}


@pytest.fixture
def api(monkeypatch):
    pipeline = StubPipeline()
    monkeypatch.setattr(main, "pipeline", pipeline)
    monkeypatch.setenv("DEMO_MOCK_MODE", "true")
    with TestClient(main.app) as client:
        yield client, pipeline


@pytest.mark.parametrize("params", [
    {"target_date": "2026-02-30"},
    {"target_date": "2026-2-14"},
    {"target_date": "../../file"},
    {"horizon_hours": 0},
    {"horizon_hours": 169},
    {"turbine_id": "missing"},
])
def test_invalid_inputs_rejected_before_pipeline(api, params):
    client, pipeline = api
    for path in ("/api/forecast", "/api/export/csv"):
        assert client.get(path, params=params).status_code == 422
    assert client.post("/api/forecast", json=params).status_code == 422
    assert pipeline.calls == []


def test_get_post_refresh_and_alias_have_same_contract(api):
    client, pipeline = api
    params = {"target_date": "2026-02-14", "horizon_hours": 24, "turbine_id": "both", "refresh_weather": True}
    via_get = client.get("/api/forecast", params=params)
    via_post = client.post("/api/forecast", json=params)
    assert via_get.status_code == via_post.status_code == 200
    assert via_get.json() == via_post.json()
    assert via_get.json()["benchmark"]["skill_score_index"] is None
    assert pipeline.calls == [("2026-02-14", 24, "farm", True)] * 2


def test_csv_contains_provenance_and_selected_turbine(api):
    client, _ = api
    response = client.get("/api/export/csv", params={"turbine_id": "turbine_2", "horizon_hours": 24})
    assert response.status_code == 200
    assert "wind_forecast_turbine_2_2026-02-14.csv" in response.headers["content-disposition"]
    frame = pd.read_csv(io.StringIO(response.text))
    assert len(frame) == 24
    assert frame["predicted_mwh"].sum() == 48.0
    assert set(frame["training_data_source"]) == {"SYNTHETIC_DEMO"}
    assert set(frame["evaluation_status"]) == {"UNVERIFIED_NO_ACTUALS"}
    assert set(frame["scenario_issue_time"]) == {"2026-02-13T23:00:00"}
    assert not frame["forecast_issue_time_verified"].any()


def test_submission_export_is_per_turbine_and_current(api):
    client, pipeline = api
    response = client.get("/api/export/submission")
    assert response.status_code == 200
    frame = pd.read_csv(io.StringIO(response.text))
    assert len(frame) == 1344
    assert len(pipeline.calls) == 56
    assert set(frame["turbine_id"]) == {"turbine_1", "turbine_2"}
    assert not frame.duplicated(["turbine_id", "timestamp"]).any()
    assert frame["predicted_mwh"].sum() == 2016.0
    assert frame.iloc[0]["scenario_issue_time"] == "2026-01-31T23:00:00"


def test_long_horizon_summary_does_not_double_count_month():
    frame, summary = build_submission(StubPipeline(), horizon_hours=48)
    assert len(frame) == 2688
    assert frame["predicted_mwh"].sum() == 4032.0
    assert summary["monthly_forecast_generation_mwh"] == 2016.0
    assert summary["monthly_accounting_turbine_hours"] == 1344
    assert summary["forecast_capacity_factor_pct"] == 60.0
    assert summary["skill_score_vs_observations"] is None
    assert summary["verified_kegoc_savings_kzt"] is None
    assert not summary["forecast_issue_time_verified"]


def test_submission_writes_requested_output_and_truthful_summary(tmp_path):
    path = tmp_path / "replay.csv"
    summary = run_test(path, pipeline=StubPipeline())
    assert len(pd.read_csv(path)) == 1344
    assert json.loads(path.with_suffix(".summary.json").read_text(encoding="utf-8")) == summary
    assert summary["weather_sources"] == ["MOCK_MODE_SYNTHESIS"]


def test_clean_error_response_and_health(api, monkeypatch):
    client, pipeline = api
    assert client.get("/api/health").json()["demo_mock_mode"] is True
    assert client.get("/api/simulation/february", params={"turbine_id": "invalid"}).status_code == 422
    assert client.get("/api/simulation/february", params={"turbine_id": "all"}).json()["turbine_id"] == "farm"

    def unavailable(**kwargs):
        raise RuntimeError("private-path-internal-detail")

    monkeypatch.setattr(pipeline, "run_forecast_cycle", unavailable)
    response = client.get("/api/forecast")
    assert response.status_code == 500
    assert "private-path" not in response.text


def test_cli_overrides_environment_and_validates_port(monkeypatch):
    calls = []
    monkeypatch.setenv("PORT", "3000")
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: calls.append(kwargs))
    main.main(["--port", "8000", "--host", "127.0.0.1"])
    assert calls == [{"host": "127.0.0.1", "port": 8000}]
    for invalid in ("0", "65536", "abc"):
        with pytest.raises(SystemExit) as exc:
            main.main(["--port", invalid])
        assert exc.value.code == 2
    assert len(calls) == 1


def test_upload_scada_valid_csv(api, tmp_path, monkeypatch):
    import backend.agent.data_loader as dl
    monkeypatch.setattr(dl, "DATA_DIR", str(tmp_path))
    client, _ = api
    dates = pd.date_range("2025-01-01", periods=24, freq="h")
    csv_content = "time,normalized_power,temperature_2m,wind_speed_10m\n" + "\n".join(
        f"{d.isoformat()},0.55,-5.0,7.2" for d in dates
    )
    files = {"file": ("test_scada.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post("/api/upload/scada", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["rows_count"] == 24
    assert data["data_source"] == "USER_SUPPLIED_UNVERIFIED"
    assert data["is_verified"] is False
    assert data["schema_validated"] is True
    assert data["mean_normalized_power"] == 0.55


def test_upload_scada_bundled_demo_is_synthetic(api, tmp_path, monkeypatch):
    import backend.agent.data_loader as dl
    monkeypatch.setattr(dl, "DATA_DIR", str(tmp_path))
    client, _ = api
    demo_file = Path("data/historical_wind.csv")
    if demo_file.is_file():
        content = demo_file.read_bytes()
        files = {"file": ("demo.csv", io.BytesIO(content), "text/csv")}
        response = client.post("/api/upload/scada", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["data_source"] == "SYNTHETIC_DEMO"
        assert data["is_verified"] is False


def test_upload_scada_training_value_error_returns_422(api, tmp_path, monkeypatch):
    import backend.agent.data_loader as dl
    monkeypatch.setattr(dl, "DATA_DIR", str(tmp_path))
    client, pipeline = api

    def failing_retrain(candidate_df, target_path=None):
        raise ValueError("Insufficient training records prior to forecast cutoff")

    monkeypatch.setattr(pipeline, "update_history_and_retrain", failing_retrain)
    dates = pd.date_range("2026-02-10", periods=5, freq="h")
    csv_content = "time,normalized_power,temperature_2m,wind_speed_10m\n" + "\n".join(
        f"{d.isoformat()},0.55,-5.0,7.2" for d in dates
    )
    files = {"file": ("short_scada.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post("/api/upload/scada", files=files)
    assert response.status_code == 422
    assert "Ошибка исторических данных для обучения" in response.json()["detail"]


def test_upload_scada_invalid_extension(api):
    client, _ = api
    files = {"file": ("test.txt", io.BytesIO(b"some,data"), "text/plain")}
    response = client.post("/api/upload/scada", files=files)
    assert response.status_code == 422


def test_upload_scada_invalid_content(api):
    client, _ = api
    files = {"file": ("test.csv", io.BytesIO(b"bad,csv,data\n1,2,3"), "text/csv")}
    response = client.post("/api/upload/scada", files=files)
    assert response.status_code == 422


def test_frontend_js_syntax_via_node():
    import subprocess
    js_test_script = Path("scripts/test_js_syntax.js")
    assert js_test_script.is_file(), "scripts/test_js_syntax.js must exist"
    res = subprocess.run(["node", str(js_test_script)], capture_output=True, text=True)
    assert "SUCCESS: Script parsed cleanly with zero syntax errors!" in res.stdout
    assert "SUCCESS: All required UI functions are defined in the script!" in res.stdout


