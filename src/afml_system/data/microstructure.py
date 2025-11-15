"""
Market microstructure features.
Implements OFI, VPIN, and Kyle's lambda from market microstructure theory.
"""
import numpy as np
import pandas as pd
from typing import Optional


def order_flow_imbalance(
    df: pd.DataFrame,
    price_col: str = 'Close',
    volume_col: str = 'Volume',
    window: int = 20
) -> pd.Series:
    """
    Calculate Order Flow Imbalance (OFI).

    OFI measures the difference between buy and sell volume, estimated
    using tick rule (price changes indicate trade direction).

    Args:
        df: DataFrame with price and volume
        price_col: Name of price column
        volume_col: Name of volume column
        window: Rolling window for aggregation

    Returns:
        Series of OFI values
    """
    prices = df[price_col]
    volumes = df[volume_col]

    # Determine tick direction
    price_changes = prices.diff()
    tick_direction = pd.Series(0, index=df.index)
    tick_direction[price_changes > 0] = 1
    tick_direction[price_changes < 0] = -1

    # Forward fill zero changes
    tick_direction = tick_direction.replace(0, np.nan).fillna(method='ffill').fillna(0)

    # Calculate signed volume
    signed_volume = tick_direction * volumes

    # Rolling OFI
    ofi = signed_volume.rolling(window).sum()

    return ofi


def vpin(
    df: pd.DataFrame,
    price_col: str = 'Close',
    volume_col: str = 'Volume',
    n_buckets: int = 50,
    window: int = 20
) -> pd.Series:
    """
    Calculate Volume-Synchronized Probability of Informed Trading (VPIN).

    VPIN estimates the probability of informed trading based on order flow imbalance.

    Args:
        df: DataFrame with price and volume
        price_col: Name of price column
        volume_col: Name of volume column
        n_buckets: Number of volume buckets
        window: Window for VPIN calculation

    Returns:
        Series of VPIN values
    """
    prices = df[price_col]
    volumes = df[volume_col]

    # Calculate volume buckets
    total_volume = volumes.sum()
    bucket_size = total_volume / n_buckets

    # Determine tick direction
    price_changes = prices.diff()
    tick_direction = pd.Series(0, index=df.index)
    tick_direction[price_changes > 0] = 1
    tick_direction[price_changes < 0] = -1
    tick_direction = tick_direction.replace(0, np.nan).fillna(method='ffill').fillna(0)

    # Calculate buy and sell volumes
    buy_volume = volumes.copy()
    buy_volume[tick_direction < 0] = 0

    sell_volume = volumes.copy()
    sell_volume[tick_direction > 0] = 0

    # Calculate VPIN
    vpin_values = []

    cumulative_volume = 0
    bucket_buy = 0
    bucket_sell = 0
    bucket_imbalances = []

    for i in range(len(df)):
        cumulative_volume += volumes.iloc[i]
        bucket_buy += buy_volume.iloc[i]
        bucket_sell += sell_volume.iloc[i]

        if cumulative_volume >= bucket_size:
            # Complete bucket
            imbalance = abs(bucket_buy - bucket_sell)
            bucket_imbalances.append(imbalance)

            # Reset
            cumulative_volume = 0
            bucket_buy = 0
            bucket_sell = 0

            # Calculate VPIN for this observation
            if len(bucket_imbalances) >= window:
                recent_imbalances = bucket_imbalances[-window:]
                total_volume_in_window = sum(recent_imbalances) * 2  # Buy + sell
                vpin_val = sum(recent_imbalances) / total_volume_in_window if total_volume_in_window > 0 else 0
                vpin_values.append(vpin_val)
            else:
                vpin_values.append(np.nan)
        else:
            vpin_values.append(np.nan)

    vpin_series = pd.Series(vpin_values, index=df.index)
    vpin_series = vpin_series.fillna(method='ffill')

    return vpin_series


def kyle_lambda(
    df: pd.DataFrame,
    price_col: str = 'Close',
    volume_col: str = 'Volume',
    window: int = 20
) -> pd.Series:
    """
    Calculate Kyle's Lambda (price impact coefficient).

    Kyle's lambda measures how much prices move per unit of order flow,
    indicating market liquidity and information content of trades.

    Args:
        df: DataFrame with price and volume
        price_col: Name of price column
        volume_col: Name of volume column
        window: Rolling window for calculation

    Returns:
        Series of Kyle's lambda values
    """
    prices = df[price_col]
    volumes = df[volume_col]

    # Calculate returns
    returns = prices.pct_change()

    # Calculate signed volume (using tick rule)
    price_changes = prices.diff()
    tick_direction = pd.Series(0, index=df.index)
    tick_direction[price_changes > 0] = 1
    tick_direction[price_changes < 0] = -1
    tick_direction = tick_direction.replace(0, np.nan).fillna(method='ffill').fillna(0)

    signed_volume = tick_direction * volumes

    # Rolling regression: returns ~ signed_volume
    lambda_values = []

    for i in range(len(df)):
        if i < window:
            lambda_values.append(np.nan)
        else:
            window_returns = returns.iloc[i-window:i]
            window_volume = signed_volume.iloc[i-window:i]

            # Remove NaN
            valid_idx = ~(window_returns.isna() | window_volume.isna())
            window_returns = window_returns[valid_idx]
            window_volume = window_volume[valid_idx]

            if len(window_returns) > 5 and window_volume.std() > 0:
                # Simple linear regression
                coef = np.cov(window_returns, window_volume)[0, 1] / (window_volume.var() + 1e-10)
                lambda_values.append(abs(coef))
            else:
                lambda_values.append(np.nan)

    lambda_series = pd.Series(lambda_values, index=df.index)
    lambda_series = lambda_series.fillna(method='ffill')

    return lambda_series


def effective_spread(
    df: pd.DataFrame,
    high_col: str = 'High',
    low_col: str = 'Low',
    close_col: str = 'Close'
) -> pd.Series:
    """
    Calculate effective spread.

    Args:
        df: DataFrame with OHLC data
        high_col: High price column
        low_col: Low price column
        close_col: Close price column

    Returns:
        Series of effective spreads
    """
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]

    # Effective spread as percentage
    spread = (high - low) / close

    return spread


def roll_spread(
    df: pd.DataFrame,
    price_col: str = 'Close',
    window: int = 20
) -> pd.Series:
    """
    Calculate Roll's spread estimator.

    Roll's spread estimates the bid-ask spread from price changes.

    Args:
        df: DataFrame with prices
        price_col: Price column name
        window: Rolling window

    Returns:
        Series of Roll's spread estimates
    """
    prices = df[price_col]
    price_changes = prices.diff()

    # Calculate covariance of successive price changes
    roll_values = []

    for i in range(len(df)):
        if i < window + 1:
            roll_values.append(np.nan)
        else:
            window_changes = price_changes.iloc[i-window:i]

            # Auto-covariance at lag 1
            cov = window_changes.iloc[1:].values @ window_changes.iloc[:-1].values / (window - 1)

            if cov < 0:
                spread = 2 * np.sqrt(-cov)
            else:
                spread = 0

            roll_values.append(spread)

    roll_series = pd.Series(roll_values, index=df.index)
    return roll_series


def amihud_illiquidity(
    df: pd.DataFrame,
    price_col: str = 'Close',
    volume_col: str = 'Volume',
    window: int = 20
) -> pd.Series:
    """
    Calculate Amihud's illiquidity measure.

    Measures price impact per dollar of trading volume.

    Args:
        df: DataFrame with price and volume
        price_col: Price column name
        volume_col: Volume column name
        window: Rolling window

    Returns:
        Series of Amihud illiquidity values
    """
    prices = df[price_col]
    volumes = df[volume_col]

    # Calculate absolute returns
    abs_returns = prices.pct_change().abs()

    # Calculate dollar volume
    dollar_volume = prices * volumes

    # Amihud measure
    illiq = abs_returns / (dollar_volume + 1e-10)

    # Rolling average
    illiq_avg = illiq.rolling(window).mean()

    return illiq_avg
