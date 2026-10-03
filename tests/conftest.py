# tests/conftest.py
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.inference import ModelInferenceEngine


@pytest.fixture
def mock_engine():
    """Provides a ModelInferenceEngine with a mocked joblib payload."""
    fake_payload = {
        "model": MagicMock(),
        "feature_cols": ["feat_1", "feat_2"],
        "optimal_threshold": 0.6,
    }
    with patch("joblib.load", return_value=fake_payload), patch.object(
        Path, "exists", return_value=True
    ):
        return ModelInferenceEngine(model_path="dummy_path.joblib")