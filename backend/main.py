"""HTTP API and single-command dashboard entry point."""

import argparse
import io
import logging
import os
from datetime import date
from pathlib import Path
import sys
from threading import Lock
from typing import Annotated, Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

load_dotenv(ROOT / ".env", override=False)
os.environ.setdefault("DEMO_MOCK_MODE", "true")

from backend.agent.pipeline import SamrukWindAgentPipeline
from backend.agent.weather_tool import TURBINES

logger = logging.getLogger(__name__)
app = FastAPI(
    title="Samruk WindAgent AI",
    description="Agentic AI for Wind Power Plant Generation Forecasting (Samruk-Kazyna - Shelek WF)",
    version="2.0.0"
)

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    from fastapi.encoders import jsonable_encoder
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

pipeline = None
_pipeline_lock = Lock()
TurbineId = Literal["turbine_1", "turbine_2", "farm", "all", "both", "cluster"]


def get_pipeline():
    """Train at most once, including concurrent requests arriving at startup."""
    global pipeline
    if pipeline is None:
        with _pipeline_lock:
            if pipeline is None:
                pipeline = SamrukWindAgentPipeline()
    return pipeline


class ForecastRequest(BaseModel):
    target_date: str = Field("2026-02-14", pattern=r"^\d{4}-\d{2}-\d{2}$", description="Start date YYYY-MM-DD")
    horizon_hours: int = Field(48, ge=1, le=168, description="Hourly steps (24 or 48 for the challenge)")
    turbine_id: TurbineId = Field("turbine_1", description="turbine_1, turbine_2, or farm")
    refresh_weather: bool = Field(False, description="Refresh weather and re-evaluate this forecast")

    @field_validator("target_date")
    @classmethod
    def validate_date(cls, value):
        date.fromisoformat(value)
        return value

    @field_validator("turbine_id")
    @classmethod
    def normalize_turbine(cls, value):
        return "farm" if value in ("all", "both", "cluster") else value


def _forecast(req: ForecastRequest):
    try:
        return get_pipeline().run_forecast_cycle(**req.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Forecast failed")
        raise HTTPException(status_code=500, detail="Forecast could not be generated. Check the server log and data source.") from exc


def _csv_response(frame: pd.DataFrame, filename: str):
    stream = io.StringIO()
    frame.to_csv(stream, index=False)
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "version": app.version,
        "model_initialized": pipeline is not None,
        "demo_mock_mode": os.environ.get("DEMO_MOCK_MODE", "true").lower() in ("true", "1", "yes"),
    }


@app.get("/api/turbines")
def get_turbines():
    return {"status": "SUCCESS", "turbines": TURBINES}


@app.post("/api/forecast")
def generate_forecast_post(req: ForecastRequest):
    return _forecast(req)


@app.get("/api/forecast")
def generate_forecast_get(req: Annotated[ForecastRequest, Depends()]):
    return _forecast(req)


@app.get("/api/simulation/february")
def get_february_simulation(turbine_id: TurbineId = Query("farm")):
    canonical_id = "farm" if turbine_id in ("all", "both", "cluster") else turbine_id
    try:
        return get_pipeline().run_full_february_simulation(turbine_id=canonical_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("February simulation failed")
        raise HTTPException(status_code=500, detail="February simulation could not be generated.") from exc


@app.get("/api/export/csv")
def export_forecast_csv(req: Annotated[ForecastRequest, Depends()]):
    result = _forecast(req)
    frame = pd.DataFrame(result["timeline"])
    benchmark = result.get("benchmark", {})
    frame["weather_source"] = benchmark.get("data_provenance", "UNKNOWN")
    frame["training_data_source"] = benchmark.get("training_data_source", "UNKNOWN")
    frame["evaluation_status"] = benchmark.get("evaluation_status", "UNVERIFIED_NO_ACTUALS")
    metadata = result.get("metadata", {})
    for field in ("turbine_id", "target_date", "timezone", "scenario_issue_time", "generated_at", "forecast_issue_time_verified"):
        frame[field] = metadata.get(field)
    return _csv_response(frame, f"wind_forecast_{req.turbine_id}_{req.target_date}.csv")


@app.get("/api/export/submission")
def export_submission_csv():
    """Return 28 daily 24-hour windows for each turbine, generated without file writes."""
    from scripts.run_february_test import build_submission

    try:
        frame, _ = build_submission(get_pipeline())
        return _csv_response(frame, "submission_forecast_february_2026.csv")
    except Exception as exc:
        logger.exception("Submission export failed")
        raise HTTPException(status_code=500, detail="February submission could not be generated.") from exc


@app.post("/api/upload/scada")
async def upload_scada_data(file: UploadFile = File(...)):
    """Upload verified SCADA historical CSV, validate schema, save and retrain models."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Только CSV файлы поддерживаются (.csv)")
    try:
        content = await file.read()
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(status_code=422, detail="Файл превышает лимит 50 МБ")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("cp1251")
        df_raw = pd.read_csv(io.StringIO(text))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Ошибка чтения CSV: {exc}") from exc

    try:
        from backend.agent.data_loader import _validate_history, DATA_DIR
        validated_df = _validate_history(df_raw)
    except ValueError as val_err:
        raise HTTPException(status_code=422, detail=f"Ошибка валидации колонок/данных SCADA: {val_err}") from val_err

    target_path = Path(DATA_DIR) / "shelek_historical.csv"
    validated_df.to_csv(target_path, index=False)

    try:
        pipe = get_pipeline()
        metadata = pipe.reload_data_and_retrain()
    except Exception as exc:
        logger.exception("Retraining failed after CSV upload")
        raise HTTPException(status_code=500, detail=f"Ошибка переобучения модели: {exc}") from exc

    speed_col = "wind_speed_100m" if "wind_speed_100m" in validated_df else "wind_speed_10m"
    return {
        "status": "SUCCESS",
        "filename": file.filename,
        "rows_count": len(validated_df),
        "start_date": validated_df["time"].iloc[0].isoformat(),
        "end_date": validated_df["time"].iloc[-1].isoformat(),
        "columns": list(validated_df.columns),
        "mean_normalized_power": round(float(validated_df["normalized_power"].mean()), 4),
        "mean_wind_speed": round(float(validated_df[speed_col].mean()), 2),
        "data_source": "USER_SUPPLIED_VERIFIED",
        "training_metadata": metadata,
        "message": "SCADA данные успешно загружены. Модели турбин переобучены и откалиброваны."
    }


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    return (ROOT / "backend" / "static" / "index.html").read_text(encoding="utf-8")


def _port(value):
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("port must be an integer from 1 to 65535") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be an integer from 1 to 65535")
    return port


def main(argv=None):
    import uvicorn

    parser = argparse.ArgumentParser(description="Start the WindAgent API and dashboard")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=_port, default=os.environ.get("PORT", "8000"))
    args = parser.parse_args(argv)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
