import os
import io
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from backend.agent.pipeline import SamrukWindAgentPipeline
from backend.agent.weather_tool import TURBINES

app = FastAPI(
    title="Samruk WindAgent AI",
    description="Agentic AI for Wind Power Plant Generation Forecasting (Samruk-Kazyna)",
    version="1.0.0"
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
def generate_forecast(req: ForecastRequest):
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

@app.get("/api/simulation/february")
def get_february_simulation():
    try:
        p = get_pipeline()
        return p.run_full_february_simulation()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/export/csv")
def export_forecast_csv(target_date: str = "2026-02-14", horizon_hours: int = 48):
    """
    Downloads forecast data as CSV formatted for Samruk-Kazyna evaluation.
    """
    p = get_pipeline()
    res = p.run_forecast_cycle(target_date=target_date, horizon_hours=horizon_hours)
    df = pd.DataFrame(res["timeline"])
    
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename=wind_forecast_{target_date}.csv"
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
    # Initialize model upon startup
    get_pipeline()
    uvicorn.run(app, host="0.0.0.0", port=8000)
