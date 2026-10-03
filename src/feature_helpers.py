"""Feature Helper Utilities — Interface Excerpts.

Selected function signatures and docstrings from the production smart hood controller.
Implementation details are intentionally omitted for IP protection.
"""

import pandas as pd


def filter_short_label_noise(
    df: pd.DataFrame,
    label_col: str = "is_cooking",
    min_duration_steps: int = 3,
) -> pd.DataFrame:
    """Filter out transient signal spikes (debouncing) on ground-truth target labels.

    Args:
        df: Input telemetry DataFrame.
        label_col: Target binary column.
        min_duration_steps: Minimum consecutive positive samples required.

    Returns:
        DataFrame with debounced label column.
    """
    df = df.copy()
    if label_col in df.columns:
        # Standard debouncing: remove positive runs shorter than min_duration_steps
        blocks = (df[label_col] != df[label_col].shift()).cumsum()
        counts = df.groupby(blocks)[label_col].transform("count")
        df[label_col] = df[label_col].where(
            (df[label_col] == 0) | (counts >= min_duration_steps), 0
        )
    return df


def create_event_target(
    df: pd.DataFrame,
    sensor_config: dict[str, float] | None = None,
    label_col: str = "is_cooking",
) -> pd.DataFrame:
    """Derive event activation targets from multi-sensor threshold triggers.

    Args:
        df: Telemetry stream DataFrame.
        sensor_config: Sensor activation thresholds.
        label_col: Target column name.

    Returns:
        DataFrame populated with derived event targets.
    """
    df = df.copy()
    if label_col not in df.columns:
        df[label_col] = 0

    if sensor_config:
        triggers = []
        for col, threshold in sensor_config.items():
            if col in df.columns:
                triggers.append(df[col] >= threshold)

        if triggers:
            # Activate target if any sensor threshold is exceeded or label is active
            sensor_trigger = pd.concat(triggers, axis=1).any(axis=1)
            df[label_col] = (df[label_col] | sensor_trigger).astype(int)

    return df
