"""Feature Engineering Pipeline — Sanitized Public Showcase.

This module demonstrates a reproducible time-series feature engineering pipeline
for smart range hood telemetry.

Production hardware mappings, proprietary signal transformations, and exact
controller thresholds are intentionally omitted for IP protection.
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

# --- DYNAMIC PROJECT ROOT RESOLUTION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import dvc.api
import mlflow
import pandas as pd

from src.feature_helpers import (
    create_event_target,
    filter_short_label_noise,
)
from src.features import engineer_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


def get_git_commit() -> str:
    """Retrieve the active Git commit hash."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
            )
            .decode("ascii")
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def get_dvc_data_version(data_path: str) -> str:
    """Retrieve the DVC-tracked data reference."""
    try:
        return dvc.api.get_url(
            path=data_path,
            repo=str(PROJECT_ROOT),
        )
    except Exception:  # noqa: BLE001
        return "local-untracked"


def get_public_sensor_configuration() -> dict[str, float]:
    """Return generic baseline thresholds for sensor targets.

    Production sensor calibration curves and proprietary threshold values
    are intentionally omitted.
    """
    return {
        "co2": 3.0,
        "voc": 3.0,
        "pm2_5": 3.0,
    }


def build_and_save_features(
    input_path: str,
    output_path: str,
) -> None:
    """Execute feature engineering pipeline and export processed dataset."""
    start_time = time.time()

    mlflow.set_experiment("smart_hood_event_detection")

    with mlflow.start_run(run_name="build_features"):
        mlflow.set_tag("git_commit", get_git_commit())
        mlflow.log_param("input_data_path", input_path)
        mlflow.log_param("output_data_path", output_path)
        mlflow.log_param(
            "dvc_data_reference",
            get_dvc_data_version(input_path),
        )

        logger.info("[1/3] Loading and normalizing timestamps...")
        df = pd.read_parquet(input_path)

        # Robust Timestamp Resolution
        if "timestamp" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if df.columns[0] != "timestamp":
                df.rename(
                    columns={df.columns[0]: "timestamp"},
                    inplace=True,
                )
        elif "timestamp" not in df.columns:
            possible_time_cols = [
                "ts",
                "datetime",
                "time",
                "Date",
                "Timestamp",
                "date",
            ]
            found_col = [col for col in possible_time_cols if col in df.columns]

            if found_col:
                df.rename(
                    columns={found_col[0]: "timestamp"},
                    inplace=True,
                )
            else:
                raise KeyError("Could not find a valid timestamp column.")

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["date_group"] = df["timestamp"].dt.date

        logger.info("[2/3] Filtering label noise & preparing event targets...")
        df = filter_short_label_noise(
            df,
            label_col="is_cooking",
        )

        df = create_event_target(
            df,
            sensor_config=get_public_sensor_configuration(),
            label_col="is_cooking",
        )

        logger.info("[3/3] Engineering telemetry features...")
        df, feature_cols = engineer_features(
            df,
            group_col="date_group",
        )

        output_file = Path(output_path)
        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_parquet(
            output_file,
            index=False,
        )

        execution_duration = time.time() - start_time

        mlflow.log_metric("engineered_feature_count", len(feature_cols))
        mlflow.log_metric("processing_duration_seconds", execution_duration)

        logger.info(
            "Feature processing complete in %.2fs.",
            execution_duration,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature engineering pipeline for smart range hood sensor telemetry."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/sample/telemetry_sample.parquet",
        help="Path to input raw telemetry parquet file",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/features_prepared.parquet",
        help="Path to output engineered features parquet file",
    )

    args = parser.parse_args()

    build_and_save_features(
        args.input,
        args.output,
    )
