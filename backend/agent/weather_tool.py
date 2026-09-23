"""Open-Meteo retrieval with explicit sources, units and demo fallback."""

import json
import os
import tempfile
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import requests


CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "weather_cache_v3")
TURBINES = {
    "turbine_1": {"lat": 43.645150, "lon": 78.535604, "name": "Turbine #1 (Shelek West)", "rated_mw": 2.5},
    "turbine_2": {"lat": 43.643198, "lon": 78.538828, "name": "Turbine #2 (Shelek East)", "rated_mw": 2.5},
    "farm": {"lat": 43.644174, "lon": 78.537216, "name": "Shelek Wind Farm (Total 2x2.5MW)", "rated_mw": 5.0},
}
WEATHER_COLS = ["temperature_2m", "wind_speed_10m", "wind_speed_100m", "wind_direction_100m", "surface_pressure"]


class WeatherAgentTool:
    """Historical retrieval is a retrospective scenario, not an issued forecast.

    Open-Meteo Historical Forecast stitches initial hours of successive model
    runs; it cannot demonstrate what was known at a historical issuance time.
    """

    def __init__(self):
        self.forecast_api_url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        self.live_api_url = "https://api.open-meteo.com/v1/forecast"
        self.archive_api_url = "https://archive-api.open-meteo.com/v1/archive"

    @staticmethod
    def _today():
        return pd.Timestamp.now(tz="Asia/Almaty").date()

    @staticmethod
    def _validate_dates(start_date, end_date):
        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
        if start.isoformat() != start_date or end.isoformat() != end_date or end < start:
            raise ValueError("Weather date range must be ordered YYYY-MM-DD dates")
        if (end - start).days > 366:
            raise ValueError("Weather request must cover at most 367 days")
        return start, end

    @staticmethod
    def _validate_frame(df, start_date, end_date):
        if not {"time", *WEATHER_COLS}.issubset(df.columns):
            raise ValueError("Weather response is missing required hourly fields")
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"], errors="raise")
        if df["time"].dt.tz is not None:
            df["time"] = df["time"].dt.tz_convert("Asia/Almaty").dt.tz_localize(None)
        expected = pd.date_range(start_date, f"{end_date} 23:00:00", freq="h")
        df = df.sort_values("time").reset_index(drop=True)
        if not pd.DatetimeIndex(df["time"]).equals(expected):
            raise ValueError("Weather response must contain every requested hour exactly once")
        df[WEATHER_COLS] = df[WEATHER_COLS].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(df[WEATHER_COLS].to_numpy(dtype=float)).all():
            raise ValueError("Weather response contains null or non-finite observations")
        if (df[["wind_speed_10m", "wind_speed_100m"]] < 0).any().any():
            raise ValueError("Wind speed must not be negative")
        if (df["surface_pressure"] <= 0).any() or (df["temperature_2m"] <= -273.15).any():
            raise ValueError("Weather pressure or temperature is physically invalid")
        return df

    @classmethod
    def _decode_weather(cls, payload, start_date, end_date):
        df = pd.DataFrame(payload.get("hourly", {}))
        units = payload.get("hourly_units", {})
        wind_factors = {"m/s": 1.0, "ms": 1.0, "km/h": 1 / 3.6, "kmh": 1 / 3.6,
                        "mph": 0.44704, "kn": 0.514444, "knots": 0.514444}
        for column in [col for col in df if col.startswith("wind_speed_")]:
            unit = units.get(column)
            if unit not in wind_factors:
                raise ValueError(f"Unknown wind unit for {column}: {unit}")
            df[column] = pd.to_numeric(df[column], errors="raise") * wind_factors[unit]
        if units.get("temperature_2m") not in ("C", "celsius", "\u00b0C"):
            raise ValueError("Weather temperature must declare Celsius units")
        if units.get("surface_pressure") not in ("hPa", "mb"):
            raise ValueError("Weather pressure must declare hPa units")
        hub_method = "native_100m"
        if "wind_speed_100m" not in df and {"wind_speed_80m", "wind_speed_120m"}.issubset(df.columns):
            df["wind_speed_100m"] = (df["wind_speed_80m"] + df["wind_speed_120m"]) / 2
            angle80 = np.radians(df["wind_direction_80m"].astype(float))
            angle120 = np.radians(df["wind_direction_120m"].astype(float))
            df["wind_direction_100m"] = np.degrees(np.arctan2(
                np.sin(angle80) + np.sin(angle120), np.cos(angle80) + np.cos(angle120)
            )) % 360
            hub_method = "linear_speed_interpolation_80m_120m; circular_direction_mean"
        df = cls._validate_frame(df, start_date, end_date)
        df.attrs["hub_wind_method"] = hub_method
        return df

    def fetch_forecast(self, lat: float, lon: float, start_date: str, end_date: str,
                       force_refresh: bool = False, storm_scenario: bool = False) -> pd.DataFrame:
        start, end = self._validate_dates(start_date, end_date)
        if not np.isfinite([lat, lon]).all() or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Invalid geographic coordinates")
        if storm_scenario:
            df = self._decode_weather(self._generate_storm_weather(start_date, end_date), start_date, end_date)
            df.attrs.update(
                data_source="STORM_TEST_SCENARIO",
                weather_kind="storm_test_scenario",
                forecast_issue_time_verified=False,
                timezone="Asia/Almaty",
                cache_hit=False,
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                fallback_reason="STORM_TEST_SCENARIO active: demonstration of wind >= 25 m/s safety cut-out",
            )
            df["data_source"] = "STORM_TEST_SCENARIO"
            return df
        mock_mode = os.environ.get("DEMO_MOCK_MODE", "").lower() in ("true", "1", "yes")
        is_historical = end < self._today()
        cache_path = os.path.join(CACHE_DIR, f"{lat:.6f}_{lon:.6f}_{start_date}_{end_date}.json")
        if not mock_mode and not force_refresh and os.path.isfile(cache_path):
            try:
                with open(cache_path, encoding="utf-8") as handle:
                    cache = json.load(handle)
                metadata = cache["metadata"]
                if cache["schema_version"] != 3 or metadata["data_source"] not in (
                    "OPEN_METEO_HISTORICAL_FORECAST", "OPEN_METEO_REANALYSIS", "OPEN_METEO_LIVE_FORECAST"
                ):
                    raise ValueError("Unverified cache provenance")
                if not is_historical:
                    age = datetime.now(timezone.utc) - datetime.fromisoformat(metadata["retrieved_at"])
                    if age.total_seconds() > 3600 or age.total_seconds() < 0:
                        raise ValueError("Live forecast cache expired")
                df = self._validate_frame(pd.DataFrame(cache["rows"]), start_date, end_date)
                df["data_source"] = metadata["data_source"]
                df.attrs.update(metadata, cache_hit=True)
                return df
            except (OSError, ValueError, TypeError, KeyError):
                pass

        failures = []
        df = None
        if not mock_mode:
            candidates = [(self.live_api_url, "OPEN_METEO_LIVE_FORECAST", "live_forecast")]
            if is_historical:
                candidates = [
                    (self.forecast_api_url, "OPEN_METEO_HISTORICAL_FORECAST", "retrospective_stitched_forecast"),
                    (self.archive_api_url, "OPEN_METEO_REANALYSIS", "retrospective_reanalysis"),
                ]
            for endpoint, source, kind in candidates:
                hourly = "temperature_2m,wind_speed_10m,wind_speed_80m,wind_speed_120m,wind_direction_80m,wind_direction_120m,surface_pressure"
                if endpoint == self.archive_api_url:
                    hourly = "temperature_2m,wind_speed_10m,wind_speed_100m,wind_direction_100m,surface_pressure"
                params = {
                    "latitude": lat, "longitude": lon, "start_date": start_date, "end_date": end_date,
                    "hourly": hourly, "wind_speed_unit": "ms", "temperature_unit": "celsius", "timezone": "Asia/Almaty",
                }
                try:
                    response = requests.get(endpoint, params=params, timeout=8)
                    response.raise_for_status()
                    df = self._decode_weather(response.json(), start_date, end_date)
                    df.attrs.update(data_source=source, weather_kind=kind)
                    break
                except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
                    failures.append(f"{source}: {type(exc).__name__}")

        if df is None:
            df = self._decode_weather(self._generate_fallback_weather(start_date, end_date), start_date, end_date)
            df.attrs.update(
                data_source="MOCK_MODE_SYNTHESIS" if mock_mode else "METEOROLOGICAL_PHYSICS_SYNTHESIS",
                weather_kind="synthetic_demo",
            )
        df.attrs.update(
            forecast_issue_time_verified=False, timezone="Asia/Almaty", cache_hit=False,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            fallback_reason="DEMO_MOCK_MODE enabled" if mock_mode else "; ".join(failures) or None,
        )
        df["data_source"] = df.attrs["data_source"]
        # Never persist synthetic fallback under a real-weather cache key.
        if df.attrs["weather_kind"] != "synthetic_demo":
            temporary_path = None
            try:
                os.makedirs(CACHE_DIR, exist_ok=True)
                rows = df.copy()
                rows["time"] = rows["time"].astype(str)
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE_DIR, delete=False) as handle:
                    temporary_path = handle.name
                    json.dump({"schema_version": 3, "metadata": df.attrs, "rows": rows.to_dict(orient="records")}, handle)
                os.replace(temporary_path, cache_path)
            except OSError:
                pass
            finally:
                if temporary_path is not None and os.path.exists(temporary_path):
                    os.unlink(temporary_path)
        return df

    def _generate_fallback_weather(self, start_date: str, end_date: str) -> dict:
        """Synthetic demonstration only; each calendar day is stable across horizons."""
        self._validate_dates(start_date, end_date)
        frames = []
        for day in pd.date_range(start_date, end_date, freq="D"):
            rng = np.random.RandomState(int(day.strftime("%Y%m%d")))
            hours = np.arange(24)
            seasonal = np.cos(2 * np.pi * (day.dayofyear - 20) / 365.25)
            wind10 = np.clip(5.5 + seasonal + 3 * np.sin(2 * np.pi * (hours - 8) / 24) + rng.normal(0, 1.8, 24), 0, 28)
            wind100 = np.clip(wind10 * 10 ** 0.20 + rng.normal(0, 0.6, 24), 0, 32)
            temperature = 12 - 18 * seasonal - 5 * np.cos(2 * np.pi * (hours - 4) / 24) + rng.normal(0, 1.5, 24)
            frames.append(pd.DataFrame({
                "time": pd.date_range(day, periods=24, freq="h").strftime("%Y-%m-%dT%H:%M"),
                "temperature_2m": np.round(temperature, 1),
                "wind_speed_10m": np.round(wind10, 2),
                "wind_speed_100m": np.round(wind100, 2),
                "wind_direction_100m": rng.randint(45, 120, 24),
                "surface_pressure": np.round(950 + rng.normal(0, 3, 24), 1),
            }))
        return {
            "hourly_units": {"wind_speed_100m": "m/s", "wind_speed_10m": "m/s", "temperature_2m": "\u00b0C", "surface_pressure": "hPa"},
            "hourly": pd.concat(frames, ignore_index=True).to_dict(orient="list"),
        }

    def _generate_storm_weather(self, start_date: str, end_date: str) -> dict:
        """Synthetic hurricane/storm scenario (wind >= 25 m/s) to demonstrate cut-out safety shutdown."""
        self._validate_dates(start_date, end_date)
        frames = []
        for day in pd.date_range(start_date, end_date, freq="D"):
            frames.append(pd.DataFrame({
                "time": pd.date_range(day, periods=24, freq="h").strftime("%Y-%m-%dT%H:%M"),
                "temperature_2m": np.full(24, -4.5),
                "wind_speed_10m": np.full(24, 21.0),
                "wind_speed_100m": np.linspace(26.5, 28.5, 24),
                "wind_direction_100m": np.full(24, 95),
                "surface_pressure": np.full(24, 940.0),
            }))
        return {
            "hourly_units": {"wind_speed_100m": "m/s", "wind_speed_10m": "m/s", "temperature_2m": "\u00b0C", "surface_pressure": "hPa"},
            "hourly": pd.concat(frames, ignore_index=True).to_dict(orient="list"),
        }

