"""src/inference.py - Production Inference Engine for Streaming Event Detection.

Demonstrates production-oriented model application, feature-schema alignment,
streaming buffer normalization, and temporal post-processing.

The proprietary feature engineering, model configuration, signal-processing
heuristics, and temporal tuning parameters are intentionally encapsulated.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Public interface to the feature-engineering stage.
# The underlying implementation is intentionally omitted.
from src.features import engineer_features


class ModelInferenceEngine:
    """Production inference engine for streaming time-series event detection."""

    def __init__(
        self,
        model_path: str = "models/event_detector.joblib",
        postprocessor: Callable[[np.ndarray, pd.Series, float], int]
        | None = None,
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError("Model artifact not available.")

        payload: dict[str, Any] = joblib.load(self.model_path)

        self.model: Any = payload["model"]
        self.expected_features: list[str] = payload["feature_cols"]
        self.optimal_threshold: float = float(
            payload.get("optimal_threshold", 0.5)
        )

        # Proprietary temporal/signal post-processing is injected
        # behind a public interface.
        self.postprocessor = postprocessor

    @staticmethod
    def _ensure_timestamp_column(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Normalize incoming telemetry to a sorted timestamp column."""
        df_clean = df.copy()

        if "timestamp" not in df_clean.columns and isinstance(
            df_clean.index, pd.DatetimeIndex
        ):
            df_clean = df_clean.reset_index()

            if df_clean.columns[0] != "timestamp":
                df_clean.rename(
                    columns={df_clean.columns[0]: "timestamp"},
                    inplace=True,
                )

        elif "timestamp" not in df_clean.columns:
            possible_time_cols = [
                "ts",
                "datetime",
                "time",
                "Date",
                "Timestamp",
                "date",
            ]

            found_col = [
                col for col in possible_time_cols if col in df_clean.columns
            ]

            if found_col:
                df_clean.rename(
                    columns={found_col[0]: "timestamp"},
                    inplace=True,
                )
            else:
                raise ValueError(
                    "DataFrame must contain a valid timestamp column."
                )

        df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])

        return df_clean.sort_values("timestamp").reset_index(drop=True)

    def _align_features(
        self,
        df_engineered: pd.DataFrame,
    ) -> pd.DataFrame:
        """Align engineered features with the model's expected schema.

        The feature definitions themselves remain encapsulated.
        """
        X = df_engineered.reindex(
            columns=self.expected_features,
            fill_value=np.nan,
        )

        float_cols = X.select_dtypes(include=["float64"]).columns

        if len(float_cols) > 0:
            X[float_cols] = X[float_cols].astype("float32")

        return X

    def _postprocess_predictions(
        self,
        probabilities: np.ndarray,
        timestamps: pd.Series,
    ) -> int:
        """Apply temporal post-processing to model predictions.

        The production filtering and signal-based gating logic is
        intentionally omitted from the public implementation.
        """
        if self.postprocessor is not None:
            return int(
                self.postprocessor(
                    probabilities,
                    timestamps,
                    self.optimal_threshold,
                )
            )

        return int(probabilities[-1] >= self.optimal_threshold)

    def predict_stream_window(
        self,
        buffer_df: pd.DataFrame,
        threshold: float | None = None,
        min_buffer_rows: int = 5,
    ) -> dict[str, int | float | str]:
        """Execute model inference over a streaming temporal buffer.

        The public implementation demonstrates the inference contract;
        proprietary feature engineering and temporal filtering remain
        encapsulated.
        """
        if threshold is None:
            threshold = self.optimal_threshold

        df_clean = self._ensure_timestamp_column(buffer_df)

        if len(df_clean) < min_buffer_rows:
            raise ValueError(
                "Insufficient temporal context: "
                f"at least {min_buffer_rows} rows are required."
            )

        # Preserve temporal context for feature engineering.
        df_clean["group_id"] = df_clean["timestamp"].dt.date

        # Feature engineering implementation is intentionally
        # abstracted from this public portfolio example.
        df_engineered, _ = engineer_features(
            df_clean,
            group_col="group_id",
        )

        # Align runtime features with the trained model schema.
        X = self._align_features(df_engineered)

        # Model probability inference.
        probabilities: np.ndarray = self.model.predict_proba(X)[:, 1]
        probabilities = np.nan_to_num(
            probabilities,
            nan=0.0,
        )

        # Temporal/signal post-processing is deliberately
        # encapsulated rather than exposing proprietary heuristics.
        prediction = self._postprocess_predictions(
            probabilities,
            df_clean["timestamp"],
        )

        latest_probability = round(
            float(probabilities[-1]),
            4,
        )

        return {
            "timestamp": str(df_clean["timestamp"].iloc[-1]),
            "event_prediction": prediction,
            "event_probability": latest_probability,
            "threshold_used": float(threshold),
        }