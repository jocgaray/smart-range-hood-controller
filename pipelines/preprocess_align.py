"""Preprocessing & Stream Alignment Pipeline — Sanitized Public Showcase.

Synchronizes continuous high-frequency telemetry streams with discrete activity annotations.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# --- DYNAMIC PROJECT ROOT RESOLUTION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.alignment import align_annotations_with_sensors
from src.preprocessing import clean_and_regularize_telemetry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


def run_preprocessing(
    sensors_path: str,
    annotations_path: str,
    output_path: str,
) -> None:
    """Clean raw telemetry, synchronize event logs, and export aligned dataset."""
    start_time = time.time()
    logger.info("[1/3] Ingesting and regularizing raw telemetry streams...")

    sensors_df = pd.read_parquet(sensors_path)

    # Resolve timestamp column
    time_cols = ["timestamp", "ts", "datetime", "time", "Date", "Timestamp"]
    found_col = next((c for c in time_cols if c in sensors_df.columns), None)

    if not found_col and isinstance(sensors_df.index, pd.DatetimeIndex):
        sensors_df = sensors_df.reset_index(names="timestamp")
        found_col = "timestamp"
    elif not found_col:
        raise KeyError("Could not locate a datetime index or valid timestamp column.")

    # Apply IoT signal cleaning & grid regularization
    cleaned_df = clean_and_regularize_telemetry(
        df=sensors_df,
        timestamp_col=found_col,
    )

    logger.info("[2/3] Synchronizing activity annotations onto sensor timeline...")
    annotations_df = pd.read_csv(annotations_path)

    aligned_df = align_annotations_with_sensors(
        sensor_df=cleaned_df.set_index("timestamp"),
        annotations_df=annotations_df,
    )

    logger.info("[3/3] Exporting aligned telemetry dataset...")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    aligned_df.reset_index(names="timestamp").to_parquet(
        output_file,
        index=False,
    )

    duration = time.time() - start_time
    logger.info(
        "Preprocessing complete (%d rows) in %.2fs -> %s",
        len(aligned_df),
        duration,
        output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Align sensor telemetry with activity event annotations."
    )

    parser.add_argument(
        "--sensors",
        type=str,
        default="data/sample/telemetry_sample.parquet",
        help="Path to raw telemetry parquet file",
    )

    parser.add_argument(
        "--annotations",
        type=str,
        default="data/sample/annotations_sample.csv",
        help="Path to raw activity annotations CSV file",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/aligned_telemetry.parquet",
        help="Path to output aligned parquet file",
    )

    args = parser.parse_args()

    run_preprocessing(
        sensors_path=args.sensors,
        annotations_path=args.annotations,
        output_path=args.output,
    )