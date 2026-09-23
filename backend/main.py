import os
import sys
import io
import pandas as pd

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from backend.agent.pipeline import SamrukWindAgentPipeline
from backend.agent.weather_tool import TURBINES

app = FastAPI(
    title="Samruk WindAgent AI",
    description="Agentic AI for Wind Power Plant Generation Forecasting (Samruk-Kazyna - Shelek Wind Farm)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (cached)
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = SamrukWindAgentPipeline()
    return pipeline

class ForecastRequest(BaseModel):
    target_date: str = "2026-02-14"
    horizon_hours: int = 48
    turbine_id: str = "turbine_1"

@app.get("/api/turbines")
def get_turbines():
    return {"status": "SUCCESS", "turbines": TURBINES}

@app.post("/api/forecast")
def generate_forecast_post(req: ForecastRequest):
    try:
        p = get_pipeline()
        result = p.run_forecast_cycle(
            target_date=req.target_date,
            horizon_hours=req.horizon_hours,
            turbine_id=req.turbine_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/forecast")
def generate_forecast_get(
    target_date: str = Query("2026-02-14", description="Start date YYYY-MM-DD"),
    horizon_hours: int = Query(48, ge=1, le=168, description="Horizon in hours (24 or 48)"),
    turbine_id: str = Query("turbine_1", description="turbine_1, turbine_2, or farm")
):
    try:
        p = get_pipeline()
        result = p.run_forecast_cycle(
            target_date=target_date,
            horizon_hours=horizon_hours,
            turbine_id=turbine_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/simulation/february")
def get_february_simulation(turbine_id: str = Query("farm", description="turbine_1, turbine_2, or farm")):
    try:
        p = get_pipeline()
        return p.run_full_february_simulation(turbine_id=turbine_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/export/csv")
def export_forecast_csv(
    target_date: str = Query("2026-02-14", description="Start date YYYY-MM-DD"),
    horizon_hours: int = Query(48, ge=1, le=168, description="Horizon in hours (24 or 48)"),
    turbine_id: str = Query("turbine_1", description="turbine_1, turbine_2, or farm")
):
    """
    Downloads forecast data as CSV formatted for Samruk-Kazyna evaluation.
    Correctly accounts for selected turbine ID and rated capacity.
    """
    p = get_pipeline()
    res = p.run_forecast_cycle(target_date=target_date, horizon_hours=horizon_hours, turbine_id=turbine_id)
    df = pd.DataFrame(res["timeline"])
    
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename=wind_forecast_{turbine_id}_{target_date}.csv"
    return response

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """
    Single-command modern interactive dashboard.
    Demonstrates the live Agentic AI loop, dispatcher insights, and generation curves.
    """
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Samruk WindAgent AI running. Dashboard loading...</h1>"

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    # Initialize model upon startup
    get_pipeline()
    uvicorn.run(app, host=host, port=port)
