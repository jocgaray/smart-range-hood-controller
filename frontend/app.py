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
except Exception:
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

# """
# frontend/app.py - Streamlit Simulation & Diagnostics Workbench.

# Demonstrates a lightweight frontend for:
# - preset simulation scenarios
# - natural-language simulation requests
# - API-driven execution
# - persistent session-state results
# - diagnostic visualization
# - metric inspection and export

# Backend simulation, physics, controller, and LLM implementation
# details remain encapsulated behind the API boundary.
# """


# import base64
# import os
# from typing import Any, Dict

# import pandas as pd
# import requests
# import streamlit as st


# API_URL = os.getenv(
#     "API_URL",
#     "http://localhost:8000/api/v1",
# )


# st.set_page_config(
#     page_title="Simulation & Diagnostics Workbench",
#     layout="wide",
# )

# st.title("Simulation & Diagnostics Workbench")


# if "sim_results" not in st.session_state:
#     st.session_state.sim_results = None


# @st.cache_data(show_spinner=False, ttl=3600)
# def fetch_scenarios_map() -> Dict[str, str]:
#     """Fetch scenarios from backend and map Title -> Key."""
#     try:
#         response = requests.get(
#             f"{API_URL}/scenarios",
#             timeout=5,
#         )
#         response.raise_for_status()

#         # Fix 1: Response is a JSON list directly
#         data = response.json()
#         if isinstance(data, list) and data:
#             return {s["title"]: s["key"] for s in data}

#     except requests.RequestException:
#         pass

#     return {
#         "High Load Scenario": "high_load",
#         "Moderate Load Scenario": "moderate_load",
#         "Low Background Scenario": "low_background",
#     }


# def execute_simulation_api(
#     endpoint: str,
#     payload: dict[str, Any],
# ) -> dict[str, Any]:
#     """Execute a request against the backend simulation API."""
#     response = requests.post(
#         f"{API_URL}/{endpoint}",
#         json=payload,
#         timeout=300,
#     )

#     response.raise_for_status()

#     return response.json()


# def send_simulation_request(
#     endpoint: str,
#     payload: dict[str, Any],
# ) -> None:
#     """Execute a simulation request and store the returned results."""
#     try:
#         with st.spinner(
#             "Running simulation and preparing diagnostics..."
#         ):
#             st.session_state.sim_results = (
#                 execute_simulation_api(
#                     endpoint,
#                     payload,
#                 )
#             )

#     except requests.exceptions.ConnectionError:
#         st.error(
#             "Unable to connect to the simulation API. "
#             "Please verify that the backend service is running."
#         )

#     except requests.exceptions.HTTPError as exc:
#         status_code = (
#             exc.response.status_code
#             if exc.response is not None
#             else "unknown"
#         )
#         st.error(
#             f"Simulation API returned an error "
#             f"(status {status_code})."
#         )

#     except requests.exceptions.RequestException:
#         st.error(
#             "The simulation request could not be completed."
#         )


# def render_simulation_dashboard() -> None:
#     """Render returned simulation results and diagnostics."""
#     results = st.session_state.sim_results

#     if not results:
#         return

#     st.markdown("---")

#     st.success("Simulation completed successfully.")

#     metrics = results.get("metrics", {})

#     # Fix 3: Wrap dict in a list so DataFrame builds cleanly
#     if metrics:
#         df_metrics = pd.DataFrame([metrics] if isinstance(metrics, dict) else metrics)
#         st.subheader(" Performance Metrics")

#         if "Controller" in df_metrics.columns:
#             df_metrics = df_metrics.set_index("Controller")

#         st.dataframe(
#             df_metrics,
#             use_container_width=True,
#         )

#         csv_data = df_metrics.to_csv().encode("utf-8")

#         st.download_button(
#             label=" Download Metrics CSV",
#             data=csv_data,
#             file_name="simulation_metrics.csv",
#             mime="text/csv",
#         )

#     plot_base64 = results.get("plot_base64")

#     if plot_base64:
#         st.subheader(" Diagnostic Visualization")

#         try:
#             plot_bytes = base64.b64decode(plot_base64)
#             st.image(
#                 plot_bytes,
#                 use_container_width=True,
#             )
#         except Exception:
#             st.warning(
#                 "Diagnostic visualization could not be displayed."
#             )


# # ---------------------------------------------------------------------
# # Scenario configuration
# # ---------------------------------------------------------------------

# st.sidebar.header(" Simulation Settings")

# scenarios_map = fetch_scenarios_map()

# selected_title = st.sidebar.selectbox(
#     "Select Scenario",
#     list(scenarios_map.keys()),
# )
# selected_key = scenarios_map[selected_title]

# sim_duration = st.sidebar.slider(
#     "Simulation Duration (minutes)",
#     min_value=10.0,
#     max_value=120.0,
#     value=60.0,
#     step=5.0,
# )


# tab_preset, tab_llm = st.tabs(
#     [
#         "Standard Presets",
#         "Natural Language",
#     ]
# )


# with tab_preset:
#     st.subheader("Run Preset Scenario")

#     st.info(
#         f"Selected scenario: **{selected_title}** "
#         f"({sim_duration:.0f} min)"
#     )

#     with st.form("preset_form"):
#         submitted = st.form_submit_button(
#             "Run Simulation",
#             type="primary",
#         )

#         if submitted:
#             # Fix 2: Send 'scenario_key' instead of 'scenario_name'
#             send_simulation_request(
#                 "simulate",
#                 {
#                     "scenario_key": selected_key,
#                     "duration_min": sim_duration,
#                 },
#             )


# with tab_llm:
#     st.subheader("Natural-Language Scenario")

#     with st.form("llm_form"):
#         prompt_text = st.text_area(
#             "Describe the scenario to simulate:",
#             value=(
#                 "Describe the desired input profile "
#                 "and its temporal evolution."
#             ),
#             height=120,
#         )

#         st.caption(
#             "The backend converts the natural-language "
#             "description into a validated simulation scenario."
#         )

#         submitted = st.form_submit_button(
#             "Parse & Simulate",
#             type="primary",
#         )

#         if submitted:
#             if not prompt_text.strip():
#                 st.warning(
#                     "Please provide a simulation scenario."
#                 )
#             else:
#                 send_simulation_request(
#                     "simulate/llm",
#                     {
#                         "prompt": prompt_text.strip(),
#                         "duration_min": sim_duration,
#                     },
#                 )


# render_simulation_dashboard()