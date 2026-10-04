"""Telemetry Signal Cleaning and Regularization Module.

Provides robust IoT time-series cleaning, bounded interpolation,
outlier clipping, and rolling-window smoothing.
"""

import numpy as np
import pandas as pd


def clean_and_regularize_telemetry(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    co2_outlier_threshold: float = 10000.0,
    max_missing_minutes: int = 15,
    smoothing_window_sec: int = 60,
) -> pd.DataFrame:
    """Clean raw IoT sensor telemetry and align onto a regular 1-second time grid.

    Args:
        df: Raw sensor telemetry DataFrame.
        timestamp_col: Name of the timestamp column.
        co2_outlier_threshold: Ceiling above which readings are treated as sensor errors.
        max_missing_minutes: Maximum gap duration allowed for nearest-neighbor imputation.
        smoothing_window_sec: Rolling mean window size in seconds for noise reduction.

    Returns:
        Regularized and sanitized Pandas DataFrame with a 'timestamp' column.
    """
    if df.empty:
        return df

    data = df.copy()

    # 1. PII Scrubbing & Schema Normalization
    pii_columns = ["Ph", "Customer", "Loc", "ID", "Phone", "Date"]
    data.drop(columns=[c for c in pii_columns if c in data.columns], inplace=True)

    # 2. Timezone Normalization & Deduplication
    data[timestamp_col] = pd.to_datetime(data[timestamp_col]).dt.tz_localize(None)
    data.drop_duplicates(subset=[timestamp_col], inplace=True)
    data.set_index(timestamp_col, inplace=True)
    data.sort_index(inplace=True)

    # 3. Create Uniform 1-Second Time Grid for Covered Span
    start_ts = data.index.min().floor("s")
    end_ts = data.index.max().ceil("s")
    full_grid = pd.date_range(start=start_ts, end=end_ts, freq="1s")

    # 4. Bounded Nearest-Neighbor Reindexing
    max_fill_samples = max_missing_minutes * 60
    regularized = data.reindex(full_grid, method="nearest", limit=max_fill_samples)

    # 5. Outlier Suppression (Static Electricity Spikes)
    co2_col = "co2" if "co2" in regularized.columns else ("CO2" if "CO2" in regularized.columns else None)
    if co2_col:
        regularized[co2_col] = regularized[co2_col].apply(
            lambda val: np.nan if val > co2_outlier_threshold else val
        )
        regularized[co2_col] = regularized[co2_col].interpolate(
            method="nearest",
            limit=max_fill_samples,
        )

    # 6. Quality & Validity Indicators
    sensor_cols = [c for c in ["temperature", "co2", "CO2", "voc", "VOC", "pm2_5", "PM2_5", "humidity"] if c in regularized.columns]
    if sensor_cols:
        regularized["is_valid_telemetry"] = regularized[sensor_cols].notna().all(axis=1).astype(int)

    # 7. Rolling Noise Reduction
    for col in sensor_cols:
        regularized[col] = (
            regularized[col]
            .rolling(window=smoothing_window_sec, min_periods=1)
            .mean()
            .round(4)
        )

    # Fill remaining post-smoothing NaNs with 0 baseline
    regularized.fillna(0.0, inplace=True)

    return regularized.reset_index(names="timestamp")