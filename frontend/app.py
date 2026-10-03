import os

import requests
import streamlit as st

# Normalize API_URL by removing trailing slashes or duplicate /api/v1 suffixes
RAW_API_URL = os.getenv("API_URL", "http://api:8000").rstrip("/")
if RAW_API_URL.endswith("/api/v1"):
    BASE_API_URL = RAW_API_URL[:-7]
else:
    BASE_API_URL = RAW_API_URL

API_V1_URL = f"{BASE_API_URL}/api/v1"

st.set_page_config(
    page_title="Simulation Workbench",
    layout="wide",
)

st.title("Simulation & Diagnostics Workbench")

# Sidebar Configuration
st.sidebar.header("Simulation Settings")

# Fetch available scenario keys from backend
try:
    scenarios_resp = requests.get(f"{API_V1_URL}/scenarios", timeout=5)
    if scenarios_resp.status_code == 200:
        scenarios_data = scenarios_resp.json()
        scenario_options = {
            s["title"]: s["key"] for s in scenarios_data
        }
    else:
        scenario_options = {"High Load Scenario": "high_load"}
except Exception:  # noqa: BLE001
    scenario_options = {"High Load Scenario": "high_load"}

selected_title = st.sidebar.selectbox(
    "Select Scenario",
    options=list(scenario_options.keys()),
)
scenario_key = scenario_options[selected_title]

duration_min = st.sidebar.slider(
    "Simulation Duration (minutes)",
    min_value=10.0,
    max_value=240.0,
    value=25.0,
    step=5.0,
)

# Main UI Tabs
tab_preset, tab_llm = st.tabs(["Standard Presets", "Natural Language"])

with tab_preset:
    st.subheader("Run Preset Scenario")
    st.info(f"Selected scenario: **{selected_title}** ({duration_min:.0f} min)")

    if st.button("Run Simulation", key="btn_preset"):
        payload = {
            "scenario_key": scenario_key,
            "duration_min": duration_min,
            "mode": "proactive",
        }

        try:
            response = requests.post(
                f"{API_V1_URL}/simulate",
                json=payload,
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                st.success(f"Simulation execution completed in {data.get('execution_time_sec')}s")
                st.json(data.get("metrics", {}))
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")

        except requests.exceptions.RequestException as err:
            st.error(f"Unable to connect to the simulation API at {API_V1_URL}. Details: {err}")

with tab_llm:
    st.subheader("Natural Language Simulation")
    prompt = st.text_area(
        "Describe simulation scenario:",
        value="Run a high-intensity activity followed by a moderate activity.",
    )

    if st.button("Run LLM Simulation", key="btn_llm"):
        payload = {
            "prompt": prompt,
            "duration_min": duration_min,
            "mode": "proactive",
        }

        try:
            response = requests.post(
                f"{API_V1_URL}/simulate/llm",
                json=payload,
                timeout=60,
            )

            if response.status_code == 200:
                data = response.json()
                st.success(f"LLM Simulation completed in {data.get('execution_time_sec')}s")
                st.json(data.get("metrics", {}))
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")

        except requests.exceptions.RequestException as err:
            st.error(f"Unable to connect to the simulation API at {API_V1_URL}. Details: {err}")