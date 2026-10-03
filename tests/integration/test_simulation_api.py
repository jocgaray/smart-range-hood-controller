"""
tests/test_simulation_api.py

Integration tests verifying FastAPI orchestration (api/main.py)
and synthetic backend engine (src/simulation_engine.py) work end-to-end.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from api.main import ActivityParams, ParsedScenarioParams, app


@pytest.fixture
def client():
    """Fixture providing a FastAPI TestClient instance."""
    return TestClient(app)


def test_health_check(client):
    """Verify that the health check endpoint returns 200 and expected service status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "simulation-api",
    }


def test_get_scenarios(client):
    """Verify that preset scenarios endpoint returns showcase scenario summaries."""
    response = client.get("/api/v1/scenarios")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 3

    keys = [scenario["key"] for scenario in data]
    assert "high_load" in keys
    assert "moderate_load" in keys
    assert "low_background" in keys


def test_simulate_preset_scenario_success(client):
    """Verify standard simulation pipeline using a valid preset scenario key."""
    payload = {
        "scenario_key": "high_load",
        "duration_min": 30.0,
        "mode": "proactive",
    }
    response = client.post("/api/v1/simulate", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert data["execution_time_sec"] >= 0.0
    assert data["metrics"]["controller"] == "Proactive"
    assert data["metrics"]["simulation_type"] == "synthetic_showcase"
    assert data["metrics"]["scenario_activities"] == 1

    # Verify telemetry structure matches public schema
    assert len(data["telemetry"]) > 0
    first_record = data["telemetry"][0]
    for required_key in (
        "time_min",
        "pm2_5",
        "co2",
        "voc",
        "event_probability",
        "actuator_state",
    ):
        assert required_key in first_record


def test_simulate_custom_params_success(client):
    """Verify standard simulation pipeline when passing inline abstract parameters."""
    custom_params = {
        "scenario_title": "Custom Test Scenario",
        "total_duration_min": 45.0,
        "activities": [
            {
                "activity_name": "Test Activity",
                "start_time_min": 2.0,
                "duration_min": 10.0,
                "intensity": "high",
            }
        ],
    }
    payload = {
        "params": custom_params,
        "duration_min": 45.0,
        "mode": "reactive",
    }
    response = client.post("/api/v1/simulate", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert data["metrics"]["controller"] == "Reactive"
    assert data["metrics"]["scenario_activities"] == 1
    assert data["parsed_params"]["scenario_title"] == "Custom Test Scenario"


def test_simulate_invalid_scenario_key_returns_404(client):
    """Verify that requesting an unregistered scenario key raises HTTP 404."""
    payload = {"scenario_key": "non_existent_key"}
    response = client.post("/api/v1/simulate", json=payload)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_simulate_invalid_mode_returns_500(client):
    """Verify HTTP 500 handling when simulation engine rejects an invalid control mode."""
    payload = {
        "scenario_key": "high_load",
        "mode": "unsupported_mode",
    }
    response = client.post("/api/v1/simulate", json=payload)
    assert response.status_code == 500
    assert response.json()["detail"] == "Simulation execution failed."


def test_simulate_llm_fallback_without_api_key(client):
    """Verify natural-language endpoint gracefully uses fallback scenario when GEMINI_API_KEY is missing."""
    with patch.dict("os.environ", {}, clear=True):
        payload = {
            "prompt": "Simulate high activity",
            "duration_min": 30.0,
            "mode": "proactive",
        }
        response = client.post("/api/v1/simulate/llm", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"
        assert (
            data["parsed_params"]["scenario_title"] == "Synthetic Moderate Load"
        )


@patch("api.main.ChatGoogleGenerativeAI")
def test_simulate_llm_success_with_mocked_llm(mock_llm_cls, client):
    """Verify natural-language simulation pipeline end-to-end with mocked LLM structured output."""
    mock_instance = MagicMock()
    mock_structured_llm = MagicMock()

    mock_llm_cls.return_value = mock_instance
    mock_instance.with_structured_output.return_value = mock_structured_llm

    mock_structured_llm.invoke.return_value = ParsedScenarioParams(
        scenario_title="Mocked LLM Activity",
        total_duration_min=20.0,
        activities=[
            ActivityParams(
                activity_name="Cooking",
                start_time_min=5.0,
                duration_min=15.0,
                intensity="high",
            )
        ],
    )

    with patch.dict("os.environ", {"GEMINI_API_KEY": "fake_test_key"}):
        payload = {
            "prompt": "Cook dinner for 15 minutes starting at minute 5.",
            "duration_min": 30.0,
            "mode": "proactive",
        }
        response = client.post("/api/v1/simulate/llm", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"
        assert data["parsed_params"]["scenario_title"] == "Mocked LLM Activity"
        assert data["metrics"]["scenario_activities"] == 1
        assert len(data["telemetry"]) > 0