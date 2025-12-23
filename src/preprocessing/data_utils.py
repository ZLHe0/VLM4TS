"""Data conversion utilities for Orion compatibility."""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


def orion_to_internal(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Orion format DataFrame to internal format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'timestamp' and 'value' columns

    Returns
    -------
    values : np.ndarray
        Time series values, shape (T,)
    timestamps : np.ndarray
        Timestamp values, shape (T,)
    """
    if 'timestamp' not in df.columns or 'value' not in df.columns:
        raise ValueError("DataFrame must have 'timestamp' and 'value' columns")

    # Sort by timestamp to ensure chronological order
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)

    values = df_sorted['value'].values
    timestamps = df_sorted['timestamp'].values

    return values, timestamps


def anomaly_scores_to_intervals(
    scores: np.ndarray,
    timestamps: np.ndarray,
    threshold: Optional[float] = None,
    percentile: float = 99.0
) -> pd.DataFrame:
    """
    Convert anomaly score vector to Orion interval format.

    Parameters
    ----------
    scores : np.ndarray
        Anomaly scores, shape (T,)
    timestamps : np.ndarray
        Timestamp values, shape (T,)
    threshold : float, optional
        Threshold for anomaly detection. If None, uses percentile.
    percentile : float, optional
        Percentile for automatic threshold (default: 99.0)

    Returns
    -------
    pd.DataFrame
        DataFrame with 'start', 'end', 'severity' columns
    """
    if threshold is None:
        threshold = np.percentile(scores, percentile)

    # Find anomalous points
    anomalous = scores > threshold

    # Find continuous intervals
    intervals = []
    in_anomaly = False
    start_idx = None

    for i in range(len(anomalous)):
        if anomalous[i] and not in_anomaly:
            # Start of anomaly
            start_idx = i
            in_anomaly = True
        elif not anomalous[i] and in_anomaly:
            # End of anomaly
            end_idx = i - 1
            severity = float(np.mean(scores[start_idx:end_idx+1]))
            intervals.append({
                'start': timestamps[start_idx],
                'end': timestamps[end_idx],
                'severity': severity
            })
            in_anomaly = False

    # Handle case where anomaly extends to end of series
    if in_anomaly:
        end_idx = len(anomalous) - 1
        severity = float(np.mean(scores[start_idx:end_idx+1]))
        intervals.append({
            'start': timestamps[start_idx],
            'end': timestamps[end_idx],
            'severity': severity
        })

    # Convert to DataFrame
    if len(intervals) == 0:
        return pd.DataFrame(columns=['start', 'end', 'severity'])

    return pd.DataFrame(intervals)


def merge_overlapping_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    """
    Merge overlapping or adjacent anomaly intervals.

    Parameters
    ----------
    intervals : pd.DataFrame
        DataFrame with 'start', 'end', 'severity' columns

    Returns
    -------
    pd.DataFrame
        Merged intervals
    """
    if len(intervals) == 0:
        return intervals

    # Sort by start time
    intervals_sorted = intervals.sort_values('start').reset_index(drop=True)

    merged = []
    current = intervals_sorted.iloc[0].to_dict()
    severities = [current['severity']]

    for i in range(1, len(intervals_sorted)):
        next_interval = intervals_sorted.iloc[i].to_dict()

        # Check if intervals overlap or are adjacent
        if next_interval['start'] <= current['end'] + 1:
            # Merge intervals
            current['end'] = max(current['end'], next_interval['end'])
            severities.append(next_interval['severity'])
            current['severity'] = np.mean(severities)
        else:
            # Save current and start new
            merged.append(current)
            current = next_interval
            severities = [current['severity']]

    # Add last interval
    merged.append(current)

    return pd.DataFrame(merged)


def intervals_from_indices(
    interval_indices: List[List[int]],
    timestamps: np.ndarray,
    scores: Optional[np.ndarray] = None,
    default_severity: float = 1.0
) -> pd.DataFrame:
    """
    Convert interval indices to Orion format with timestamps.

    Parameters
    ----------
    interval_indices : list of [start_idx, end_idx] pairs
        Anomaly intervals as index pairs
    timestamps : np.ndarray
        Timestamp values for the time series
    scores : np.ndarray, optional
        Anomaly scores to compute severity. If None, uses default_severity.
    default_severity : float, optional
        Default severity value (default: 1.0)

    Returns
    -------
    pd.DataFrame
        DataFrame with 'start', 'end', 'severity' columns
    """
    if len(interval_indices) == 0:
        return pd.DataFrame(columns=['start', 'end', 'severity'])

    intervals = []
    for start_idx, end_idx in interval_indices:
        # Clamp indices
        start_idx = max(0, min(start_idx, len(timestamps) - 1))
        end_idx = max(0, min(end_idx, len(timestamps) - 1))

        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        # Compute severity
        if scores is not None:
            severity = float(np.mean(scores[start_idx:end_idx+1]))
        else:
            severity = default_severity

        intervals.append({
            'start': timestamps[start_idx],
            'end': timestamps[end_idx],
            'severity': severity
        })

    return pd.DataFrame(intervals)
