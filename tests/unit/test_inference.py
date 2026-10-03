# tests/unit/test_inference.py
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.inference import ModelInferenceEngine

# --- 1. Initialization Tests ---

def test_init_raises_file_not_found():
    """Verify initialization raises FileNotFoundError if model_path does not exist."""
    with pytest.raises(FileNotFoundError, match="Model artifact not available"):
        ModelInferenceEngine(model_path="non_existent_model.joblib")


def test_init_loads_payload_correctly():
    """Verify payload parameters are assigned correctly on initialization."""
    fake_payload = {
        "model": "dummy_model_object",
        "feature_cols": ["feat_a", "feat_b"],
        "optimal_threshold": 0.75,
    }
    with patch("joblib.load", return_value=fake_payload), patch.object(
        Path, "exists", return_value=True
    ):
        engine = ModelInferenceEngine(model_path="dummy.joblib")

        assert engine.model == "dummy_model_object"
        assert engine.expected_features == ["feat_a", "feat_b"]
        assert engine.optimal_threshold == 0.75


# --- 2. Helper Method Tests ---

def test_ensure_timestamp_from_index(mock_engine):
    """Verify DatetimeIndex is converted into a sorted timestamp column."""
    df_raw = pd.DataFrame(
        {"temp": [22.0, 21.5]},
        index=pd.to_datetime(["2026-01-01 10:01:00", "2026-01-01 10:00:00"]),
    )

    df_clean = mock_engine._ensure_timestamp_column(df_raw)

    assert "timestamp" in df_clean.columns
    assert df_clean["timestamp"].iloc[0] == pd.Timestamp("2026-01-01 10:00:00")


def test_ensure_timestamp_alias_mapping(mock_engine):
    """Verify time column aliases (e.g., 'ts') are renamed to 'timestamp'."""
    df_raw = pd.DataFrame(
        {
            "ts": ["2026-01-01 10:01:00", "2026-01-01 10:00:00"],
            "temp": [22.0, 21.5],
        }
    )

    df_clean = mock_engine._ensure_timestamp_column(df_raw)

    assert "timestamp" in df_clean.columns
    assert pd.api.types.is_datetime64_any_dtype(df_clean["timestamp"])
    assert df_clean["timestamp"].iloc[0] == pd.Timestamp("2026-01-01 10:00:00")


def test_ensure_timestamp_missing_raises(mock_engine):
    """Verify ValueError is raised when no timestamp column or alias is present."""
    df_invalid = pd.DataFrame({"temp": [22.0, 21.5], "val": [1, 2]})

    with pytest.raises(ValueError, match="DataFrame must contain a valid timestamp column"):
        mock_engine._ensure_timestamp_column(df_invalid)


def test_align_features_schema_and_casting(mock_engine):
    """Verify missing features are padded with NaN and float64 is cast to float32."""
    df_engineered = pd.DataFrame(
        {
            "feat_1": np.array([1.5, 2.5], dtype="float64"),
            "extra_col": [99, 100],
        }
    )

    X_aligned = mock_engine._align_features(df_engineered)

    assert list(X_aligned.columns) == ["feat_1", "feat_2"]
    assert X_aligned["feat_2"].isna().all()
    assert X_aligned["feat_1"].dtype == np.float32


# --- 3. Post-processing Tests ---

def test_postprocess_predictions_default_fallback(mock_engine):
    """Verify postprocessing uses thresholding when no custom postprocessor is set."""
    mock_engine.postprocessor = None
    probs = np.array([0.2, 0.7])
    timestamps = pd.Series(pd.to_datetime(["2026-01-01 10:00:00", "2026-01-01 10:01:00"]))

    # 0.7 >= 0.6 threshold -> 1
    pred_high = mock_engine._postprocess_predictions(probs, timestamps)
    assert pred_high == 1

    # 0.4 < 0.6 threshold -> 0
    pred_low = mock_engine._postprocess_predictions(np.array([0.2, 0.4]), timestamps)
    assert pred_low == 0


def test_postprocess_predictions_custom_callable(mock_engine):
    """Verify custom postprocessor object is invoked when present."""
    mock_postprocessor = MagicMock(return_value=1)
    mock_engine.postprocessor = mock_postprocessor

    probs = np.array([0.1, 0.3])
    timestamps = pd.Series(pd.to_datetime(["2026-01-01 10:00:00", "2026-01-01 10:01:00"]))

    result = mock_engine._postprocess_predictions(probs, timestamps)

    mock_postprocessor.assert_called_once_with(probs, timestamps, 0.6)
    assert result == 1


# --- 4. Validation Guardrail Test ---

def test_predict_stream_window_insufficient_rows(mock_engine):
    """Verify ValueError is raised if buffer contains fewer than min_buffer_rows."""
    short_buffer = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="1min"),
            "val": [1, 2, 3],
        }
    )

    with pytest.raises(ValueError, match="Insufficient temporal context"):
        mock_engine.predict_stream_window(short_buffer, min_buffer_rows=5)