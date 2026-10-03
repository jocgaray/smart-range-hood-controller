"""Telemetry Feature Engineering Module — Sanitized Public Showcase.

This module provides time-series feature extraction pipelines for air quality
and environmental sensor telemetry.

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
    """Compute first-order temporal derivatives (differences) for sensor signals."""
    feature_cols = []
    for col in sensor_cols:
        feature_name = f"{col}_diff_1m"
        df[feature_name] = df[col].diff().fillna(0.0)
        feature_cols.append(feature_name)
    return df, feature_cols


def _extract_rolling_statistics(
    df: pd.DataFrame,
    sensor_cols: list[str],
    windows: list[int] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute multi-scale rolling means and standard deviations."""
    if windows is None:
        windows = [3, 5, 15]

    feature_cols = []
    for window in windows:
        for col in sensor_cols:
            mean_name = f"{col}_roll_mean_{window}m"
            std_name = f"{col}_roll_std_{window}m"

            rolling_obj = df[col].rolling(window=window, min_periods=1)
            df[mean_name] = rolling_obj.mean().fillna(0.0)
            df[std_name] = rolling_obj.std().fillna(0.0)

            feature_cols.extend([mean_name, std_name])
    return df, feature_cols


def _extract_lag_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    lags: list[int] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Extract historical sensor value lags for temporal context."""
    if lags is None:
        lags = [1, 3]

    feature_cols = []
    for lag in lags:
        for col in sensor_cols:
            feature_name = f"{col}_lag_{lag}m"
            # Avoid backfilling (bfill) to prevent temporal data leakage
            df[feature_name] = df[col].shift(lag).fillna(0.0)
            feature_cols.append(feature_name)
    return df, feature_cols


def engineer_features(
    df: pd.DataFrame,
    group_col: str = "date_group",
    sensor_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Engineer baseline time-series features from raw telemetry streams.

    Args:
        df: Input DataFrame containing timestamped sensor telemetry.
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
