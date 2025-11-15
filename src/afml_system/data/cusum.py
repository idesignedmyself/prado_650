"""
CUSUM filter for event detection.
Implements the Cumulative Sum filter from Advances in Financial Machine Learning.
"""
import numpy as np
import pandas as pd
from typing import Optional


def cusum_filter(
    prices: pd.Series,
    threshold: float,
    timestamps: Optional[pd.DatetimeIndex] = None
) -> pd.DatetimeIndex:
    """
    CUSUM filter for detecting significant price movements.

    The CUSUM filter triggers events when cumulative deviations from the mean
    exceed a threshold, indicating a significant directional move.

    Args:
        prices: Price series
        threshold: Threshold for event detection (e.g., 0.01 for 1%)
        timestamps: Optional timestamps (uses prices.index if None)

    Returns:
        DatetimeIndex of event timestamps
    """
    if timestamps is None:
        timestamps = prices.index

    # Calculate returns
    returns = prices.pct_change().fillna(0)

    # Initialize
    events = []
    s_pos = 0  # Positive CUSUM
    s_neg = 0  # Negative CUSUM

    for i in range(len(returns)):
        # Update CUSUM
        s_pos = max(0, s_pos + returns.iloc[i])
        s_neg = min(0, s_neg + returns.iloc[i])

        # Check for events
        if s_pos > threshold:
            events.append(timestamps[i])
            s_pos = 0
            s_neg = 0
        elif s_neg < -threshold:
            events.append(timestamps[i])
            s_pos = 0
            s_neg = 0

    return pd.DatetimeIndex(events)


def cusum_filter_symmetric(
    prices: pd.Series,
    threshold: float
) -> pd.DataFrame:
    """
    Symmetric CUSUM filter that tracks both up and down moves.

    Args:
        prices: Price series
        threshold: Threshold for event detection

    Returns:
        DataFrame with event times and directions
    """
    returns = prices.pct_change().fillna(0)

    events = []
    s_pos = 0
    s_neg = 0

    for i in range(len(returns)):
        s_pos = max(0, s_pos + returns.iloc[i])
        s_neg = min(0, s_neg + returns.iloc[i])

        if s_pos > threshold:
            events.append({
                'timestamp': prices.index[i],
                'direction': 1,
                'magnitude': s_pos
            })
            s_pos = 0
            s_neg = 0
        elif s_neg < -threshold:
            events.append({
                'timestamp': prices.index[i],
                'direction': -1,
                'magnitude': abs(s_neg)
            })
            s_pos = 0
            s_neg = 0

    return pd.DataFrame(events)


def adaptive_cusum_filter(
    prices: pd.Series,
    base_threshold: float,
    volatility_window: int = 20
) -> pd.DatetimeIndex:
    """
    Adaptive CUSUM filter with volatility-adjusted threshold.

    Args:
        prices: Price series
        base_threshold: Base threshold (multiplied by volatility)
        volatility_window: Window for volatility calculation

    Returns:
        DatetimeIndex of event timestamps
    """
    returns = prices.pct_change().fillna(0)
    volatility = returns.rolling(volatility_window).std()

    events = []
    s_pos = 0
    s_neg = 0

    for i in range(volatility_window, len(returns)):
        # Adaptive threshold
        vol = volatility.iloc[i] if not pd.isna(volatility.iloc[i]) else base_threshold
        threshold = base_threshold * vol

        # Update CUSUM
        s_pos = max(0, s_pos + returns.iloc[i])
        s_neg = min(0, s_neg + returns.iloc[i])

        # Check for events
        if s_pos > threshold:
            events.append(prices.index[i])
            s_pos = 0
            s_neg = 0
        elif s_neg < -threshold:
            events.append(prices.index[i])
            s_pos = 0
            s_neg = 0

    return pd.DatetimeIndex(events)


def cusum_events_with_side(
    prices: pd.Series,
    threshold: float
) -> pd.DataFrame:
    """
    CUSUM filter that returns events with side information.

    Args:
        prices: Price series
        threshold: Threshold for event detection

    Returns:
        DataFrame with columns: timestamp, side (1/-1)
    """
    returns = prices.pct_change().fillna(0)

    events = []
    s_pos = 0
    s_neg = 0

    for i in range(len(returns)):
        s_pos = max(0, s_pos + returns.iloc[i])
        s_neg = min(0, s_neg + returns.iloc[i])

        if s_pos > threshold:
            events.append({
                'timestamp': prices.index[i],
                'side': 1
            })
            s_pos = 0
            s_neg = 0
        elif s_neg < -threshold:
            events.append({
                'timestamp': prices.index[i],
                'side': -1
            })
            s_pos = 0
            s_neg = 0

    df = pd.DataFrame(events)
    if not df.empty:
        df.set_index('timestamp', inplace=True)

    return df
