import os
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .physics import WindTurbinePhysics

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "wind_model.joblib")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

FEATURE_COLS = [
    "wind_speed_10m",
    "wind_speed_100m",
    "temperature_2m",
    "air_density",
    "theoretical_power",
    "wind_cube",
    "sin_hour",
    "cos_hour"
]

class WindForecastingModel:
    """
    Physics-Informed Gradient Boosting Model for Wind Power Plant Generation Forecasting.
    Combines aerodynamic power curve priors with LightGBM non-linear error correction.
    """
    def __init__(self):
        self.model = LGBMRegressor(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            random_state=42,
            verbose=-1
        )
        self.is_trained = False

    def train(self, train_df: pd.DataFrame, target_col: str = "normalized_power") -> dict:
        """
        Trains the model on historical operational data.
        """
        df_feat = WindTurbinePhysics.enrich_features(train_df)
        
        # Guarantee all feature columns exist
        for col in FEATURE_COLS:
            if col not in df_feat.columns:
                df_feat[col] = 0.0

        X = df_feat[FEATURE_COLS].fillna(0)
        y = np.clip(train_df[target_col].values, 0.0, 1.0)

        # Train-val split (last 15% as validation)
        split_idx = int(len(X) * 0.85)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Validation metrics
        preds_val = np.clip(self.model.predict(X_val), 0.0, 1.0)
        rmse = float(np.sqrt(mean_squared_error(y_val, preds_val)))
        mae = float(mean_absolute_error(y_val, preds_val))
        r2 = float(r2_score(y_val, preds_val))

        # Save model
        joblib.dump(self.model, MODEL_PATH)

        return {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2_score": round(r2, 4),
            "train_samples": len(X_train),
            "val_samples": len(X_val)
        }

    def predict(self, weather_df: pd.DataFrame) -> np.ndarray:
        """
        Produces hourly normalized power predictions (0.0 to 1.0).
        """
        df_feat = WindTurbinePhysics.enrich_features(weather_df)
        for col in FEATURE_COLS:
            if col not in df_feat.columns:
                df_feat[col] = 0.0

        X = df_feat[FEATURE_COLS].fillna(0)

        if not self.is_trained:
            if os.path.exists(MODEL_PATH):
                self.model = joblib.load(MODEL_PATH)
                self.is_trained = True
            else:
                # Fallback to pure physical aerodynamic power curve if model not yet trained
                return df_feat["theoretical_power"].values

        raw_preds = self.model.predict(X)
        
        # Physical boundary clipping & storm cut-off guarantee
        preds = np.clip(raw_preds, 0.0, 1.0)
        
        # Hard physical override: if wind >= 25 m/s, turbine MUST cut out to 0
        storm_mask = df_feat["wind_speed_100m"].values >= WindTurbinePhysics.V_CUT_OUT
        preds[storm_mask] = 0.0

        # Hard physical override: if wind < 2.5 m/s, turbine cannot generate
        calm_mask = df_feat["wind_speed_100m"].values < 2.5
        preds[calm_mask] = 0.0

        return np.round(preds, 4)
