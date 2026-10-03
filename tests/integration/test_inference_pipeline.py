# tests/integration/test_inference_pipeline.py
from unittest.mock import patch
import numpy as np
import pandas as pd


def test_predict_stream_window_end_to_end_contract(mock_engine):
    """Verify full predict_stream_window pipeline contract execution."""
    buffer_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 00:00:00", periods=6, freq="1min"),
            "sensor_reading": [10.0, 12.0, 15.0, 20.0, 25.0, 30.0],
        }
    )

    df_engineered_mock = pd.DataFrame(
        {
            "feat_1": [1.0] * 6,
            "feat_2": [2.0] * 6,
        }
    )

    # Mock probabilities returned by model.predict_proba
    mock_engine.model.predict_proba.return_value = np.array(
        [[0.1, 0.1], [0.1, 0.2], [0.1, 0.3], [0.1, 0.4], [0.1, 0.5], [0.1, 0.85]]
    )

    with patch("src.inference.engineer_features", return_value=(df_engineered_mock, None)):
        result = mock_engine.predict_stream_window(buffer_df, min_buffer_rows=5)

    # Verify return dictionary contract
    assert result == {
        "timestamp": "2026-01-01 00:05:00",
        "event_prediction": 1,
        "event_probability": 0.85,
        "threshold_used": 0.6,
    }