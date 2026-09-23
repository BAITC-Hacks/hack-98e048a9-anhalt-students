import numpy as np
import pandas as pd

class WindTurbinePhysics:
    """
    Generic turbine approximation with wind shear and dry-air density adjustment.

    The cubic curve is a demo prior, not a manufacturer curve or IEC certification.
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
        t_kelvin = np.asarray(temperature_c, dtype=float) + 273.15
        if not np.all(np.isfinite(t_kelvin)) or np.any(t_kelvin <= 0):
            raise ValueError("Temperature must be finite and above absolute zero")
        if pressure_hpa is None:
            pressure_pa = 101325.0 # default sea-level/nominal
        else:
            pressure_pa = np.asarray(pressure_hpa, dtype=float) * 100.0
        if not np.all(np.isfinite(pressure_pa)) or np.any(pressure_pa <= 0):
            raise ValueError("Pressure must be finite and positive (hPa)")
        
        rho = pressure_pa / (cls.R_SPECIFIC * t_kelvin)
        return rho

    @classmethod
    def theoretical_power_curve(cls, wind_speed_hub: np.ndarray, air_density: np.ndarray = None) -> np.ndarray:
        """
        Calculates normalized active power (0.0 to 1.0) according to aerodynamic power curve.
        Accounts for density scaling and hard cut-in / cut-out thresholds.
        """
        wind = np.maximum(np.asarray(wind_speed_hub, dtype=float), 0.0)
        if not np.all(np.isfinite(wind)):
            raise ValueError("Wind speed must be finite")
        if air_density is None:
            density_ratio = 1.0
        else:
            density_ratio = np.broadcast_to(np.asarray(air_density, dtype=float), wind.shape) / cls.RHO_STANDARD
            if not np.all(np.isfinite(density_ratio)) or np.any(density_ratio <= 0):
                raise ValueError("Air density must be finite and positive")

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
            v_hub = res["wind_speed_100m"].to_numpy(dtype=float)
        elif "wind_speed_10m" in res.columns:
            v_hub = res["wind_speed_10m"].values * ((cls.HUB_HEIGHT / 10.0) ** 0.20)
            res["wind_speed_100m"] = v_hub
        else:
            raise ValueError("Weather must include wind_speed_100m or wind_speed_10m")
        if "wind_speed_10m" not in res.columns:
            res["wind_speed_10m"] = v_hub / ((cls.HUB_HEIGHT / 10.0) ** 0.20)

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
        else:
            res["sin_hour"] = 0.0
            res["cos_hour"] = 1.0

        return res
