import pandas as pd


def _infer_activity_end_time(
    start_time: pd.Timestamp,
    label: str,
    sensor_df: pd.DataFrame,
) -> pd.Timestamp:
    """
    Infer missing activity boundaries from post-event sensor dynamics.

    The implementation is intentionally omitted from this public example.
    """
    return start_time


def align_annotations_with_sensors(
    sensor_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map interval-based activity annotations onto a continuous
    time-series sensor timeline.
    """

    # Input validation
    if sensor_df.empty or annotations_df.empty:
        raise ValueError("Input dataframes must not be empty.")

    if not isinstance(sensor_df.index, pd.DatetimeIndex):
        raise TypeError("sensor_df must use a DatetimeIndex.")

    df = sensor_df.copy()
    df["household_activity"] = "Occupied_Passive"

    # Process annotations chronologically for deterministic labeling
    annotations = annotations_df.sort_values("ts")

    # Map activity intervals onto sensor observations
    for _, annotation in annotations.iterrows():
        label = str(annotation["Label"]).strip()
        start_time = pd.to_datetime(annotation["ts"])
        end_time = annotation.get("end_ts", pd.NaT)

        # Handle annotations with missing end timestamps
        if pd.isna(end_time):
            end_time = _infer_activity_end_time(
                start_time,
                label,
                df,
            )
        else:
            end_time = pd.to_datetime(end_time)

        # Assign the activity label to observations within the interval
        mask = (df.index >= start_time) & (df.index <= end_time)

        if mask.any():
            df.loc[mask, "household_activity"] = label

    return df