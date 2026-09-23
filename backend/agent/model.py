"""Chronologically evaluated LightGBM with explicit training provenance."""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .physics import WindTurbinePhysics


FEATURE_COLS = [
    "wind_speed_10m", "wind_speed_100m", "temperature_2m", "air_density",
    "theoretical_power", "wind_cube", "sin_hour", "cos_hour",
]


class WindForecastingModel:
    """Physics features supply a prior; validation is not a field accuracy claim."""

    def __init__(self):
        self.model = LGBMRegressor(
            n_estimators=150, learning_rate=0.05, num_leaves=31,
            max_depth=6, random_state=42, verbose=-1, n_jobs=2,
        )
        self.is_trained = False
        self.training_metadata = {}

    @staticmethod
    def _features(df):
        enriched = WindTurbinePhysics.enrich_features(df)
        if "temperature_2m" not in enriched:
            raise ValueError("Training and forecast weather require temperature_2m")
        features = enriched[FEATURE_COLS].astype(float)
        if not np.isfinite(features.to_numpy()).all():
            raise ValueError("Model features contain missing or non-finite values")
        return enriched, features

    @staticmethod
    def _physical_bounds(predictions, enriched):
        predictions = np.clip(np.asarray(predictions, dtype=float), 0.0, 1.0)
        wind = enriched["wind_speed_100m"].to_numpy(dtype=float)
        predictions[(wind < WindTurbinePhysics.V_CUT_IN) |
                    (wind >= WindTurbinePhysics.V_CUT_OUT)] = 0.0
        return predictions

    def train(self, train_df: pd.DataFrame, target_col: str = "normalized_power", cutoff=None) -> dict:
        """Hold out the last 15% chronologically, then refit only eligible history.

        ``cutoff`` is exclusive. Callers forecasting earlier dates must filter or
        retrain with that cutoff; a saved model of unknown vintage is never loaded.
        """
        if "time" not in train_df or target_col not in train_df:
            raise ValueError("Historical data require time and normalized power")
        history = train_df.copy()
        history["time"] = pd.to_datetime(history["time"], errors="raise")
        if history["time"].isna().any():
            raise ValueError("Historical timestamps must not be missing")
        if history["time"].dt.tz is not None:
            history["time"] = history["time"].dt.tz_convert("Asia/Almaty").dt.tz_localize(None)
        history = history.sort_values("time", kind="stable").reset_index(drop=True)
        if cutoff is not None:
            boundary = pd.Timestamp(cutoff)
            if boundary.tzinfo is not None:
                boundary = boundary.tz_convert("Asia/Almaty").tz_localize(None)
            history = history.loc[history["time"] < boundary].reset_index(drop=True)
        if history["time"].duplicated().any():
            raise ValueError("Historical data must contain one record per timestamp")
        if len(history) < 10:
            raise ValueError("At least 10 historical records before the forecast origin are required")

        enriched, features = self._features(history)
        target = history[target_col].to_numpy(dtype=float)
        if not np.isfinite(target).all() or np.any((target < 0) | (target > 1)):
            raise ValueError("normalized_power must be finite and within [0, 1]")

        split_idx = min(int(len(features) * 0.85), len(features) - 2)
        self.model.fit(features.iloc[:split_idx], target[:split_idx])
        validation = self._physical_bounds(
            self.model.predict(features.iloc[split_idx:]), enriched.iloc[split_idx:]
        )
        actual = target[split_idx:]
        metrics = {
            "rmse": round(float(np.sqrt(mean_squared_error(actual, validation))), 4),
            "mae": round(float(mean_absolute_error(actual, validation)), 4),
            "r2_score": round(float(r2_score(actual, validation)), 4),
            "train_samples": split_idx,
            "val_samples": len(actual),
            "fitted_samples": len(history),
            "train_start": history["time"].iloc[0].isoformat(),
            "train_end": history["time"].iloc[-1].isoformat(),
            "validation_train_end": history["time"].iloc[split_idx - 1].isoformat(),
            "validation_start": history["time"].iloc[split_idx].isoformat(),
            "validation_end": history["time"].iloc[-1].isoformat(),
            "data_source": train_df.attrs.get("data_source", "UNVERIFIED_INPUT"),
            "is_synthetic": train_df.attrs.get("is_synthetic"),
            "is_verified": train_df.attrs.get("is_verified", False),
            "evaluation_scope": "Chronological holdout on training source; not a 24-48h operational backtest",
        }
        # Validation metrics above belong to the holdout fit, not this final refit.
        self.model.fit(features, target)
        self.is_trained = True
        self.training_metadata = metrics
        return metrics

    def predict(self, weather_df: pd.DataFrame) -> np.ndarray:
        """Return bounded hourly normalized power; untrained models use the prior."""
        enriched, features = self._features(weather_df)
        if features.empty:
            return np.array([], dtype=float)
        raw = self.model.predict(features) if self.is_trained else enriched["theoretical_power"].to_numpy()
        return np.round(self._physical_bounds(raw, enriched), 4)
