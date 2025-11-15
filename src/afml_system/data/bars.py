"""
Information-driven bars: dollar bars, volume bars, and volatility bars.
Implements sampling methods from Advances in Financial Machine Learning.
"""
import numpy as np
import pandas as pd
from typing import Optional


def dollar_bars(
    df: pd.DataFrame,
    threshold: float,
    price_col: str = 'Close',
    volume_col: str = 'Volume'
) -> pd.DataFrame:
    """
    Generate dollar bars.

    Dollar bars are formed when the cumulative dollar value traded exceeds a threshold.

    Args:
        df: DataFrame with OHLCV data
        threshold: Dollar threshold for bar formation
        price_col: Name of price column
        volume_col: Name of volume column

    Returns:
        DataFrame with dollar bars
    """
    # Calculate dollar volume
    dollar_volume = df[price_col] * df[volume_col]

    bars = []
    cumulative = 0
    bar_data = []

    for idx, row in df.iterrows():
        bar_data.append(row)
        cumulative += dollar_volume.loc[idx]

        if cumulative >= threshold:
            # Create bar
            bar_df = pd.DataFrame(bar_data)
            bar = {
                'Open': bar_df[price_col].iloc[0],
                'High': bar_df[price_col].max(),
                'Low': bar_df[price_col].min(),
                'Close': bar_df[price_col].iloc[-1],
                'Volume': bar_df[volume_col].sum(),
                'timestamp': idx
            }
            bars.append(bar)

            # Reset
            cumulative = 0
            bar_data = []

    result = pd.DataFrame(bars)
    if not result.empty:
        result.set_index('timestamp', inplace=True)

    return result


def volume_bars(
    df: pd.DataFrame,
    threshold: float,
    price_col: str = 'Close',
    volume_col: str = 'Volume'
) -> pd.DataFrame:
    """
    Generate volume bars.

    Volume bars are formed when cumulative volume exceeds a threshold.

    Args:
        df: DataFrame with OHLCV data
        threshold: Volume threshold for bar formation
        price_col: Name of price column
        volume_col: Name of volume column

    Returns:
        DataFrame with volume bars
    """
    bars = []
    cumulative = 0
    bar_data = []

    for idx, row in df.iterrows():
        bar_data.append(row)
        cumulative += row[volume_col]

        if cumulative >= threshold:
            # Create bar
            bar_df = pd.DataFrame(bar_data)
            bar = {
                'Open': bar_df[price_col].iloc[0],
                'High': bar_df[price_col].max(),
                'Low': bar_df[price_col].min(),
                'Close': bar_df[price_col].iloc[-1],
                'Volume': bar_df[volume_col].sum(),
                'timestamp': idx
            }
            bars.append(bar)

            # Reset
            cumulative = 0
            bar_data = []

    result = pd.DataFrame(bars)
    if not result.empty:
        result.set_index('timestamp', inplace=True)

    return result


def volatility_bars(
    df: pd.DataFrame,
    threshold: float,
    price_col: str = 'Close',
    volume_col: str = 'Volume'
) -> pd.DataFrame:
    """
    Generate volatility bars.

    Volatility bars are formed when cumulative volatility (sum of absolute returns)
    exceeds a threshold.

    Args:
        df: DataFrame with OHLCV data
        threshold: Volatility threshold for bar formation
        price_col: Name of price column
        volume_col: Name of volume column

    Returns:
        DataFrame with volatility bars
    """
    bars = []
    cumulative = 0
    bar_data = []
    prev_price = None

    for idx, row in df.iterrows():
        bar_data.append(row)

        # Calculate volatility contribution
        if prev_price is not None:
            ret = abs((row[price_col] - prev_price) / prev_price)
            cumulative += ret

        prev_price = row[price_col]

        if cumulative >= threshold and len(bar_data) > 1:
            # Create bar
            bar_df = pd.DataFrame(bar_data)
            bar = {
                'Open': bar_df[price_col].iloc[0],
                'High': bar_df[price_col].max(),
                'Low': bar_df[price_col].min(),
                'Close': bar_df[price_col].iloc[-1],
                'Volume': bar_df[volume_col].sum(),
                'timestamp': idx
            }
            bars.append(bar)

            # Reset
            cumulative = 0
            bar_data = []
            prev_price = None

    result = pd.DataFrame(bars)
    if not result.empty:
        result.set_index('timestamp', inplace=True)

    return result


def imbalance_bars(
    df: pd.DataFrame,
    threshold: float,
    price_col: str = 'Close',
    volume_col: str = 'Volume'
) -> pd.DataFrame:
    """
    Generate tick imbalance bars.

    Bars are formed when cumulative signed volume imbalance exceeds threshold.

    Args:
        df: DataFrame with tick data
        threshold: Imbalance threshold
        price_col: Name of price column
        volume_col: Name of volume column

    Returns:
        DataFrame with imbalance bars
    """
    bars = []
    cumulative_imbalance = 0
    bar_data = []
    prev_price = None

    for idx, row in df.iterrows():
        bar_data.append(row)

        # Determine tick direction
        if prev_price is not None:
            if row[price_col] > prev_price:
                tick_sign = 1
            elif row[price_col] < prev_price:
                tick_sign = -1
            else:
                tick_sign = 0
        else:
            tick_sign = 0

        # Update imbalance
        cumulative_imbalance += tick_sign * row[volume_col]

        prev_price = row[price_col]

        if abs(cumulative_imbalance) >= threshold and len(bar_data) > 1:
            # Create bar
            bar_df = pd.DataFrame(bar_data)
            bar = {
                'Open': bar_df[price_col].iloc[0],
                'High': bar_df[price_col].max(),
                'Low': bar_df[price_col].min(),
                'Close': bar_df[price_col].iloc[-1],
                'Volume': bar_df[volume_col].sum(),
                'timestamp': idx
            }
            bars.append(bar)

            # Reset
            cumulative_imbalance = 0
            bar_data = []

    result = pd.DataFrame(bars)
    if not result.empty:
        result.set_index('timestamp', inplace=True)

    return result


def get_optimal_bar_threshold(
    df: pd.DataFrame,
    bar_type: str = 'dollar',
    target_bars: int = 100
) -> float:
    """
    Estimate optimal threshold for desired number of bars.

    Args:
        df: Input DataFrame
        bar_type: Type of bars ('dollar', 'volume', 'volatility')
        target_bars: Desired number of bars

    Returns:
        Estimated threshold
    """
    if bar_type == 'dollar':
        total = (df['Close'] * df['Volume']).sum()
    elif bar_type == 'volume':
        total = df['Volume'].sum()
    elif bar_type == 'volatility':
        total = df['Close'].pct_change().abs().sum()
    else:
        raise ValueError(f"Unknown bar type: {bar_type}")

    threshold = total / target_bars
    return threshold
