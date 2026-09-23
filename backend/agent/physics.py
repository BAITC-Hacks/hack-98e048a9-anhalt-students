import numpy as np
import pandas as pd

class WindTurbinePhysics:
    """
    Physical Aerodynamic and Meteorological models for Industrial Wind Turbines (Shelek WF).
    Incorporates IEC 61400-12 power curves, Hellman wind shear, and air density adjustments.
    """
    # Standard industrial turbine specs (e.g. Goldwind / Vestas 2.5 - 3.3 MW)
    V_CUT_IN = 3.0      # m/s
    V_RATED = 12.5     # m/s
    V_CUT_OUT = 25.0   # m/s (Emergency storm shut-off)
    RHO_STANDARD = 1.225 # kg/m^3 (ISA standard air density)
    R_SPECIFIC = 287.058 # J/(kg*K) gas constant for dry air
    HUB_HEIGHT = 100.0  # meters

    @classmethod
    def calculate_air_density(cls, temperature_c: np.ndarray, pressure_hpa: np.ndarray = None) -> np.ndarray:
        """
        Calculates real-time air density rho = P / (R * T).
        Winter cold in Kazakhstan (-15C) increases air density by >10%, drastically boosting power output.
        """
        t_kelvin = np.maximum(temperature_c + 273.15, 220.0)
        if pressure_hpa is None:
            pressure_pa = 101325.0 # default sea-level/nominal
        else:
            pressure_pa = pressure_hpa * 100.0
        
        rho = pressure_pa / (cls.R_SPECIFIC * t_kelvin)
        return np.clip(rho, 1.0, 1.45)

    @classmethod
    def theoretical_power_curve(cls, wind_speed_hub: np.ndarray, air_density: np.ndarray = None) -> np.ndarray:
        """
        Calculates normalized active power (0.0 to 1.0) according to aerodynamic power curve.
        Accounts for density scaling and hard cut-in / cut-out thresholds.
        """
        wind = np.maximum(wind_speed_hub, 0.0)
        if air_density is None:
            density_ratio = 1.0
        else:
            density_ratio = air_density / cls.RHO_STANDARD

        # Vectorized power calculation
        p_norm = np.zeros_like(wind, dtype=float)

        # Region 2: Between cut-in and rated (Cubic ramp)
        mask_ramp = (wind >= cls.V_CUT_IN) & (wind < cls.V_RATED)
        fraction = (wind[mask_ramp] - cls.V_CUT_IN) / (cls.V_RATED - cls.V_CUT_IN)
        if isinstance(density_ratio, np.ndarray):
            p_norm[mask_ramp] = (fraction ** 3) * density_ratio[mask_ramp]
        else:
            p_norm[mask_ramp] = (fraction ** 3) * density_ratio

        # Region 3: Rated power plateau
        mask_rated = (wind >= cls.V_RATED) & (wind < cls.V_CUT_OUT)
        p_norm[mask_rated] = 1.0

        # Region 4: Storm cut-out (P = 0)
        mask_cutout = (wind >= cls.V_CUT_OUT)
        p_norm[mask_cutout] = 0.0

        return np.clip(p_norm, 0.0, 1.0)

    @classmethod
    def enrich_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches meteorological dataframe with physical turbine features.
        """
        res = df.copy()

        # Wind speed at hub height (if missing, use Hellman shear law alpha=0.20)
        if "wind_speed_100m" in res.columns:
            v_hub = res["wind_speed_100m"].values
        elif "wind_speed_10m" in res.columns:
            v_hub = res["wind_speed_10m"].values * ((cls.HUB_HEIGHT / 10.0) ** 0.20)
            res["wind_speed_100m"] = v_hub
        else:
            v_hub = np.full(len(res), 6.0)

        temp = res["temperature_2m"].values if "temperature_2m" in res.columns else np.zeros(len(res))
        press = res["surface_pressure"].values if "surface_pressure" in res.columns else np.full(len(res), 1013.25)

        rho = cls.calculate_air_density(temp, press)
        p_phys = cls.theoretical_power_curve(v_hub, rho)

        res["air_density"] = np.round(rho, 4)
        res["theoretical_power"] = np.round(p_phys, 4)

        # Kinetic energy flux proportional to rho * v^3
        res["wind_cube"] = np.round((v_hub ** 3) * (rho / cls.RHO_STANDARD), 2)

        # Cyclic temporal features (Hour of day, day of week)
        if "time" in res.columns:
            if not pd.api.types.is_datetime64_any_dtype(res["time"]):
                res["time"] = pd.to_datetime(res["time"])
            res["hour"] = res["time"].dt.hour
            res["dayofweek"] = res["time"].dt.dayofweek
            res["sin_hour"] = np.sin(2 * np.pi * res["hour"] / 24.0)
            res["cos_hour"] = np.cos(2 * np.pi * res["hour"] / 24.0)

        return res
