"""Telemetry Feature Engineering Module — Sanitized Public Showcase.

This module provides time-series feature extraction pipelines for air quality
and environmental sensor telemetry assuming a constant 1Hz sampling rate.

Note:
    The public implementation intentionally uses generic statistical signal-processing
    primitives (rolling windows, rate-of-change, lag features) rather than reproducing
    the proprietary production feature pipeline.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Default illustrative sensor channels for public sample telemetry
DEFAULT_SENSOR_COLS: list[str] = ["co2", "voc", "pm2_5", "temperature", "humidity"]


def _extract_rate_of_change(
    df: pd.DataFrame, sensor_cols: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Compute 1-minute rate of change (first-order temporal derivative per second) for 1Hz telemetry."""
    feature_cols = []
    for col in sensor_cols:
        feature_name = f"{col}_diff_1m"
        # At 1Hz, looking 60 rows back represents a 60-second span.
        # Dividing by 60.0 scales the total difference into a per-second rate.
        df[feature_name] = df[col].diff(periods=60).fillna(0.0) / 60.0
        feature_cols.append(feature_name)
    return df, feature_cols


def _extract_rolling_statistics(
    df: pd.DataFrame,
    sensor_cols: list[str],
    windows_minutes: list[int] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute multi-scale rolling means and standard deviations for 1Hz telemetry (windows specified in minutes)."""
    if windows_minutes is None:
        windows_minutes = [3, 5, 15]

    feature_cols = []
    for win_min in windows_minutes:
        # Convert minute-based window to exact row count for 1Hz sampling (1 min = 60 rows)
        row_window = win_min * 60

        for col in sensor_cols:
            mean_name = f"{col}_roll_mean_{win_min}m"
            std_name = f"{col}_roll_std_{win_min}m"

            rolling_obj = df[col].rolling(window=row_window, min_periods=1)
            df[mean_name] = rolling_obj.mean().fillna(0.0)
            df[std_name] = rolling_obj.std().fillna(0.0)

            feature_cols.extend([mean_name, std_name])
    return df, feature_cols


def _extract_lag_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    lags_minutes: list[int] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Extract historical sensor value lags for temporal context (lags specified in minutes)."""
    if lags_minutes is None:
        lags_minutes = [1, 3]

    feature_cols = []
    for lag_min in lags_minutes:
        # Convert minute-based lag to exact row count for 1Hz sampling (1 min = 60 rows)
        row_lag = lag_min * 60

        for col in sensor_cols:
            feature_name = f"{col}_lag_{lag_min}m"
            # Shift back by the exact number of rows corresponding to the minute lag
            df[feature_name] = df[col].shift(row_lag).fillna(0.0)
            feature_cols.append(feature_name)
    return df, feature_cols


def engineer_features(
    df: pd.DataFrame,
    group_col: str = "date_group",
    sensor_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Engineer baseline time-series features from raw 1Hz telemetry streams.

    Args:
        df: Input DataFrame containing 1Hz timestamped sensor telemetry.
        group_col: Column name used to group processing sessions/days.
        sensor_cols: Optional list of target sensor column names.

    Returns:
        Tuple containing:
            - Processed DataFrame populated with engineered feature columns.
            - List of engineered feature column names.

    Raises:
        ValueError: If no valid sensor columns are found in the input DataFrame.
    """
    df = df.copy()

    # Explicit sensor column selection
    if sensor_cols is None:
        sensor_cols = [c for c in DEFAULT_SENSOR_COLS if c in df.columns]

    if not sensor_cols:
        raise ValueError(
            f"No supported sensor columns found in DataFrame. "
            f"Expected one of {DEFAULT_SENSOR_COLS}, but found {df.columns.tolist()}"
        )

    logger.info(
        "Extracting statistical telemetry features across signals: %s", sensor_cols
    )

    # Execute feature generation per group session to prevent cross-session leakage
    if group_col in df.columns:
        grouped_dfs = []
        for _, group_df in df.groupby(group_col, sort=False):
            group_df, diff_cols = _extract_rate_of_change(group_df, sensor_cols)
            group_df, roll_cols = _extract_rolling_statistics(group_df, sensor_cols)
            group_df, lag_cols = _extract_lag_features(group_df, sensor_cols)
            grouped_dfs.append(group_df)

        df = pd.concat(grouped_dfs, axis=0).reset_index(drop=True)
        all_engineered_cols = list(dict.fromkeys(diff_cols + roll_cols + lag_cols))
    else:
        logger.warning(
            "Group column '%s' not found. Engineering features across full dataset.",
            group_col,
        )
        df, diff_cols = _extract_rate_of_change(df, sensor_cols)
        df, roll_cols = _extract_rolling_statistics(df, sensor_cols)
        df, lag_cols = _extract_lag_features(df, sensor_cols)
        all_engineered_cols = list(dict.fromkeys(diff_cols + roll_cols + lag_cols))

    logger.info(
        "Successfully engineered %d telemetry features.", len(all_engineered_cols)
    )
    return df, all_engineered_cols
