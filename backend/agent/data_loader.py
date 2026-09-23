import os
import hashlib
import numpy as np
import pandas as pd
from .physics import WindTurbinePhysics

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)
HISTORICAL_CSV = os.path.join(DATA_DIR, "historical_wind.csv")
# Hash uses LF newlines so git's Windows newline conversion cannot change provenance.
BUNDLED_DEMO_SHA256 = "36345475014c34f367ca87b48c53883c8e91b210713b1cb57821895409cf05e7"


def _validate_history(df: pd.DataFrame) -> pd.DataFrame:
    required = {"time", "normalized_power", "temperature_2m"}
    if not required.issubset(df.columns) or not {"wind_speed_10m", "wind_speed_100m"}.intersection(df.columns):
        raise ValueError("Historical CSV requires time, normalized_power, temperature_2m and wind_speed_10m or wind_speed_100m")
    df["time"] = pd.to_datetime(df["time"], errors="raise")
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_convert("Asia/Almaty").dt.tz_localize(None)
    if df.empty or df["time"].isna().any() or df["time"].duplicated().any():
        raise ValueError("Historical CSV must have nonempty, unique hourly timestamps")
    numeric = [name for name in ("normalized_power", "temperature_2m", "wind_speed_10m", "wind_speed_100m", "surface_pressure") if name in df]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(df[numeric].to_numpy(dtype=float)).all() or not df["normalized_power"].between(0, 1).all():
        raise ValueError("Historical values must be finite; normalized_power must lie within [0, 1]")
    return df.sort_values("time", kind="stable").reset_index(drop=True)

def get_or_create_historical_data() -> pd.DataFrame:
    """
    Load supplied history or explicitly labelled synthetic demo history.

    A user CSV takes precedence over the bundled example. No file is implicitly
    authenticated as metered generation; provenance travels in DataFrame attrs.
    """
    # Check if a user-supplied CSV exists in data/ or task_data/
    candidates = [
        os.path.join(DATA_DIR, "shelek_historical.csv"),
        os.path.join(os.path.dirname(__file__), "..", "..", "task_data", "sheet1.csv"),
        HISTORICAL_CSV,
    ]
    for c in candidates:
        if os.path.isfile(c):
            df = _validate_history(pd.read_csv(c))
            with open(c, "rb") as source_file:
                digest = hashlib.sha256(source_file.read().replace(b"\r\n", b"\n")).hexdigest()
            synthetic = digest == BUNDLED_DEMO_SHA256 or (
                "data_source" in df and df["data_source"].eq("SYNTHETIC_DEMO").all()
            )
            df.attrs.update({
                "data_source": "SYNTHETIC_DEMO" if synthetic else "USER_SUPPLIED_UNVERIFIED",
                "is_synthetic": True if synthetic else None,
                "is_verified": False,
                "source_file": os.path.basename(c),
            })
            return df

    # Generate physically consistent dataset for March 2023 - January 31, 2026 (~25,500 hourly records)
    print("[DataLoader] Generating SYNTHETIC DEMO history; this is not metered generation.")
    date_range = pd.date_range(start="2023-03-01 00:00:00", end="2026-01-31 23:00:00", freq="h")
    n = len(date_range)
    rng = np.random.RandomState(42)

    # Seasonal variation: stronger winds in winter/spring
    day_of_year = date_range.dayofyear.values
    hour_of_day = date_range.hour.values

    # Base wind speed in Shelek corridor (annual mean ~7.2 m/s at 10m, ~9.5 m/s at 100m)
    season_factor = 1.0 + 0.25 * np.cos(2 * np.pi * (day_of_year - 40) / 365.25)
    diurnal_factor = 1.0 + 0.20 * np.sin(2 * np.pi * (hour_of_day - 9) / 24.0)

    # Weibull wind distribution
    weibull_base = rng.weibull(2.1, n) * 6.8
    v_10m = np.clip(weibull_base * season_factor * diurnal_factor, 0.2, 28.0)

    # Wind at 100m hub height (alpha ~ 0.21)
    v_100m = np.clip(v_10m * ((100.0 / 10.0) ** 0.21) + rng.normal(0, 0.6, n), 0.5, 32.0)

    # Temperature: seasonal curve (-15C in Jan to +32C in July)
    mean_temp = 12.0 - 18.0 * np.cos(2 * np.pi * (day_of_year - 20) / 365.25)
    temp_noise = rng.normal(0, 3.5, n)
    t_2m = np.round(mean_temp + temp_noise, 1)

    # Surface pressure
    p_surf = np.round(950.0 - 5.0 * np.sin(day_of_year / 20) + rng.normal(0, 4, n), 1)

    # Physical power calculation with real-world noise & efficiency loss
    rho = WindTurbinePhysics.calculate_air_density(t_2m, p_surf)
    theo_power = WindTurbinePhysics.theoretical_power_curve(v_100m, rho)

    # Add realistic operational variance (wake effect, yaw error, micro-turbulence)
    turb_noise = rng.normal(0, 0.035, n)
    # 2% chance of maintenance / curtailment event
    maint_mask = rng.rand(n) < 0.015
    p_active = np.clip(theo_power + turb_noise, 0.0, 1.0)
    p_active[maint_mask] *= rng.uniform(0.0, 0.5, np.sum(maint_mask))
    p_active[(v_100m < WindTurbinePhysics.V_CUT_IN) | (v_100m >= WindTurbinePhysics.V_CUT_OUT)] = 0.0
    p_active = np.round(p_active, 4)

    df_hist = pd.DataFrame({
        "time": date_range,
        "wind_speed_10m": np.round(v_10m, 2),
        "wind_speed_100m": np.round(v_100m, 2),
        "temperature_2m": t_2m,
        "surface_pressure": p_surf,
        "normalized_power": p_active,
        "data_source": "SYNTHETIC_DEMO",
    })

    df_hist.to_csv(HISTORICAL_CSV, index=False)
    df_hist.attrs.update({"data_source": "SYNTHETIC_DEMO", "is_synthetic": True, "is_verified": False, "source_file": os.path.basename(HISTORICAL_CSV)})
    print(f"[DataLoader] Successfully saved {len(df_hist)} records to {HISTORICAL_CSV}")
    return df_hist
