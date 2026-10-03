"""
api/main.py

FastAPI Orchestration Layer — Public Showcase.

Demonstrates an end-to-end natural-language-to-simulation API using:

    natural language
        -> structured LLM output
        -> Pydantic validation
        -> deterministic scenario reconciliation
        -> simulation engine
        -> metrics and telemetry

Production physical model parameters, calibration data, proprietary
scenario definitions, controller logic, and detailed simulation
configuration are intentionally omitted for IP protection.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.simulation_engine import SimulationEngine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


# ==========================================================
# 1. STRUCTURED OUTPUT SCHEMAS
# ==========================================================


class ActivityParams(BaseModel):
    """Structured representation of a generic simulation activity."""

    activity_name: str = Field(
        ...,
        description="Name of the simulated activity",
    )

    start_time_min: float = Field(
        ...,
        ge=0.0,
        description="Activity start offset in minutes",
    )

    duration_min: float = Field(
        ...,
        gt=0.0,
        description="Activity duration in minutes",
    )

    intensity: str = Field(
        default="medium",
        description="Generic activity intensity level",
    )


class ParsedScenarioParams(BaseModel):
    """Validated scenario representation produced by the LLM."""

    scenario_title: str = Field(
        ...,
        description="Brief descriptive scenario title",
    )

    total_duration_min: float = Field(
        ...,
        ge=0.0,
        description="Total active timeline in minutes",
    )

    activities: List[ActivityParams] = Field(
        default_factory=list,
    )


class LLMSimulationRequest(BaseModel):
    """API request for natural-language simulation."""

    prompt: str = Field(
        ...,
        min_length=1,
        json_schema_extra={
            "example": "Run a high-intensity activity followed by a moderate activity."
        },
    )

    duration_min: float = Field(
        default=60.0,
        ge=10.0,
        le=240.0,
    )

    mode: str = Field(
        default="proactive",
        description="Control mode: 'proactive' or 'reactive'",
    )


class StandardSimulationRequest(BaseModel):
    """API request for a preset or abstract scenario."""

    scenario_key: Optional[str] = Field(
        default="high_load",
    )

    params: Optional[Dict[str, Any]] = Field(
        default=None,
    )

    duration_min: float = Field(
        default=60.0,
        gt=0.0,
    )

    mode: str = Field(
        default="proactive",
    )


class ScenarioSummary(BaseModel):
    """Public summary of an available showcase scenario."""

    key: str
    title: str
    description: str


class SimulationResponse(BaseModel):
    """Standardized simulation API response."""

    status: str
    execution_time_sec: float
    parsed_params: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any]
    telemetry: List[Dict[str, Any]]
    plot_base64: str


# ==========================================================
# 2. PUBLIC SHOWCASE SCENARIOS
# ==========================================================

# These scenarios intentionally use abstract parameters.
#
# They are showcase configurations only and do not reproduce
# production physical source terms, calibration values, or
# proprietary simulation settings.

PRESET_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "high_load": {
        "title": "High Load Scenario",
        "description": (
            "Synthetic high-intensity activity used to exercise "
            "the simulation and controller pipeline."
        ),
        "params": {
            "scenario_title": "Synthetic High Load",
            "total_duration_min": 60.0,
            "activities": [
                {
                    "activity_name": "High-Intensity Activity",
                    "start_time_min": 5.0,
                    "duration_min": 15.0,
                    "intensity": "high",
                }
            ],
        },
    },
    "moderate_load": {
        "title": "Moderate Load Scenario",
        "description": (
            "Synthetic moderate-intensity activity for "
            "baseline controller evaluation."
        ),
        "params": {
            "scenario_title": "Synthetic Moderate Load",
            "total_duration_min": 60.0,
            "activities": [
                {
                    "activity_name": "Moderate Activity",
                    "start_time_min": 10.0,
                    "duration_min": 25.0,
                    "intensity": "medium",
                }
            ],
        },
    },
    "low_background": {
        "title": "Low Background Scenario",
        "description": (
            "Synthetic low-intensity background condition "
            "for sensitivity testing."
        ),
        "params": {
            "scenario_title": "Synthetic Background",
            "total_duration_min": 60.0,
            "activities": [
                {
                    "activity_name": "Background Activity",
                    "start_time_min": 0.0,
                    "duration_min": 60.0,
                    "intensity": "low",
                }
            ],
        },
    },
}


# ==========================================================
# 3. LLM PARSING & DETERMINISTIC RECONCILIATION
# ==========================================================


def reconcile_scenario_parameters(
    scenario: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Apply deterministic consistency rules to a structured scenario.

    Physical model parameters and production-specific reconciliation
    logic are intentionally omitted from this public implementation.
    """

    activities = scenario.get("activities", [])

    if not activities:
        return scenario

    max_end_time = max(
        activity["start_time_min"] + activity["duration_min"]
        for activity in activities
    )

    scenario["total_duration_min"] = max(
        scenario.get("total_duration_min", 0.0),
        max_end_time,
    )

    return scenario


def parse_and_validate_prompt(prompt: str) -> Dict[str, Any]:
    """
    Convert natural-language input into a validated abstract scenario.

    Uses Gemini structured output followed by deterministic
    scenario reconciliation.
    """

    gemini_key = os.getenv("GEMINI_API_KEY")

    if not gemini_key:
        logger.warning(
            "GEMINI_API_KEY missing. "
            "Falling back to the public showcase scenario."
        )

        return PRESET_SCENARIOS["moderate_load"]["params"]

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("LLM_MODEL", "gemini-3.8-flash"),
        google_api_key=gemini_key,
        temperature=0,
    )

    structured_llm = llm.with_structured_output(
        ParsedScenarioParams
    )

    system_prompt = (
        "Convert the user's natural-language scenario into "
        "structured activities with timing, duration, and "
        "generic intensity information. "
        "Return values compatible with the provided schema. "
        "Do not invent physical calibration parameters."
    )

    parsed: ParsedScenarioParams = structured_llm.invoke(
        [
            ("system", system_prompt),
            ("user", prompt),
        ]
    )

    extracted = parsed.model_dump()

    # Deterministic reconciliation is kept public because it
    # demonstrates how structured LLM output is made internally
    # consistent before entering the simulation layer.
    extracted = reconcile_scenario_parameters(extracted)

    return extracted


# ==========================================================
# 4. FASTAPI ORCHESTRATION ROUTER
# ==========================================================


app = FastAPI(
    title="Simulation Orchestration API",
    version="1.0.0",
    description=(
        "REST API for scenario querying, natural-language "
        "parsing, and simulation execution."
    ),
)


@app.get(
    "/health",
    status_code=status.HTTP_200_OK,
)
def health_check() -> Dict[str, str]:
    """Return service health status."""

    return {
        "status": "healthy",
        "service": "simulation-api",
    }


@app.get(
    "/api/v1/scenarios",
    response_model=List[ScenarioSummary],
)
def get_scenarios() -> List[ScenarioSummary]:
    """Return configured public showcase scenarios."""

    return [
        ScenarioSummary(
            key=key,
            title=data["title"],
            description=data["description"],
        )
        for key, data in PRESET_SCENARIOS.items()
    ]


@app.post(
    "/api/v1/simulate",
    response_model=SimulationResponse,
)
def run_standard_simulation(
    req: StandardSimulationRequest,
) -> SimulationResponse:
    """Execute a preset or abstract scenario."""

    t_start = time.perf_counter()

    if req.params:
        scenario_params = req.params

    elif req.scenario_key in PRESET_SCENARIOS:
        scenario_params = PRESET_SCENARIOS[
            req.scenario_key
        ]["params"]

    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Scenario key '{req.scenario_key}' "
                "not found."
            ),
        )

    try:
        res = SimulationEngine.run_simulation(
            params=scenario_params,
            duration_min=req.duration_min,
            mode=req.mode,
        )

    except Exception as err:
        logger.error(
            "Simulation failure: %s",
            str(err),
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Simulation execution failed.",
        )

    return SimulationResponse(
        status="success",
        execution_time_sec=round(
            time.perf_counter() - t_start,
            3,
        ),
        parsed_params=scenario_params,
        metrics=res.metrics,
        telemetry=res.telemetry.to_dict(
            orient="records",
        ),
        plot_base64=res.plot_base64,
    )


@app.post(
    "/api/v1/simulate/llm",
    response_model=SimulationResponse,
)
def run_llm_simulation(
    req: LLMSimulationRequest,
) -> SimulationResponse:
    """Execute the end-to-end natural-language simulation pipeline."""

    t_start = time.perf_counter()

    try:
        # 1. Convert natural language into a structured scenario.
        parsed_params = parse_and_validate_prompt(
            req.prompt,
        )

        # 2. Determine a sufficient simulation horizon.
        total_active_time = parsed_params.get(
            "total_duration_min",
            0.0,
        )

        sim_duration = max(
            req.duration_min,
            total_active_time + 15.0,
        )

        # 3. Dispatch the validated scenario to the
        #    simulation engine.
        res = SimulationEngine.run_simulation(
            params=parsed_params,
            duration_min=sim_duration,
            mode=req.mode,
        )

        return SimulationResponse(
            status="success",
            execution_time_sec=round(
                time.perf_counter() - t_start,
                3,
            ),
            parsed_params=parsed_params,
            metrics=res.metrics,
            telemetry=res.telemetry.to_dict(
                orient="records",
            ),
            plot_base64=res.plot_base64,
        )

    except Exception as err:
        logger.error(
            "LLM simulation failure: %s",
            str(err),
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Simulation orchestration failed.",
        )