"""
Microstructure-based features.
Transforms microstructure indicators into features for ML.
"""
import numpy as np
import pandas as pd
from ..data.microstructure import (
    order_flow_imbalance,
    vpin,
    kyle_lambda,
    effective_spread,
    roll_spread,
    amihud_illiquidity
)


def get_microstructure_features(
    df: pd.DataFrame,
    windows: list[int] = [5, 10, 20]
) -> pd.DataFrame:
    """
    Create comprehensive microstructure feature set.

    Args:
        df: DataFrame with OHLCV data
        windows: Windows for rolling calculations

    Returns:
        DataFrame with microstructure features
    """
    features = pd.DataFrame(index=df.index)

    # Order Flow Imbalance
    for window in windows:
        features[f'ofi_{window}'] = order_flow_imbalance(df, window=window)

    # VPIN
    features['vpin'] = vpin(df, n_buckets=50, window=20)

    # Kyle's Lambda
    for window in windows:
        features[f'kyle_lambda_{window}'] = kyle_lambda(df, window=window)

    # Spread measures
    features['effective_spread'] = effective_spread(df)
    features['roll_spread'] = roll_spread(df, window=20)

    # Illiquidity
    for window in windows:
        features[f'amihud_illiq_{window}'] = amihud_illiquidity(df, window=window)

    # Volume features
    features['volume_ma_20'] = df['Volume'].rolling(20).mean()
    features['volume_std_20'] = df['Volume'].rolling(20).std()
    features['volume_ratio'] = df['Volume'] / features['volume_ma_20']

    # Price impact
    returns = df['Close'].pct_change()
    for window in windows:
        features[f'price_impact_{window}'] = (
            returns.abs().rolling(window).mean() /
            (df['Volume'].rolling(window).mean() + 1e-10)
        )

    # Trade intensity
    for window in windows:
        features[f'trade_intensity_{window}'] = df['Volume'].rolling(window).sum()

    return features


def get_ofi_features(
    df: pd.DataFrame,
    windows: list[int] = [5, 10, 20, 50]
) -> pd.DataFrame:
    """
    Order Flow Imbalance features at multiple horizons.

    Args:
        df: DataFrame with OHLCV data
        windows: Windows for OFI calculation

    Returns:
        DataFrame with OFI features
    """
    features = pd.DataFrame(index=df.index)

    for window in windows:
        ofi = order_flow_imbalance(df, window=window)
        features[f'ofi_{window}'] = ofi

        # Normalized OFI
        ofi_std = ofi.rolling(window * 2).std()
        features[f'ofi_norm_{window}'] = ofi / (ofi_std + 1e-10)

        # OFI changes
        features[f'ofi_change_{window}'] = ofi.diff()

    return features


def get_vpin_features(
    df: pd.DataFrame,
    n_buckets_list: list[int] = [25, 50, 100]
) -> pd.DataFrame:
    """
    VPIN features with different bucket sizes.

    Args:
        df: DataFrame with OHLCV data
        n_buckets_list: List of bucket counts

    Returns:
        DataFrame with VPIN features
    """
    features = pd.DataFrame(index=df.index)

    for n_buckets in n_buckets_list:
        vpin_val = vpin(df, n_buckets=n_buckets, window=20)
        features[f'vpin_{n_buckets}'] = vpin_val

        # VPIN changes
        features[f'vpin_change_{n_buckets}'] = vpin_val.diff()

        # VPIN percentile
        features[f'vpin_pct_{n_buckets}'] = vpin_val.rolling(100).rank(pct=True)

    return features


def get_liquidity_features(
    df: pd.DataFrame,
    windows: list[int] = [10, 20, 50]
) -> pd.DataFrame:
    """
    Liquidity-based features.

    Args:
        df: DataFrame with OHLCV data
        windows: Windows for calculation

    Returns:
        DataFrame with liquidity features
    """
    features = pd.DataFrame(index=df.index)

    # Spread features
    features['effective_spread'] = effective_spread(df)
    features['roll_spread'] = roll_spread(df)

    for window in windows:
        # Amihud illiquidity
        features[f'amihud_{window}'] = amihud_illiquidity(df, window=window)

        # Relative spread
        spread = (df['High'] - df['Low']) / df['Close']
        features[f'rel_spread_{window}'] = spread.rolling(window).mean()

        # Volume volatility
        features[f'volume_vol_{window}'] = df['Volume'].rolling(window).std()

    return features


def get_price_impact_features(
    df: pd.DataFrame,
    windows: list[int] = [5, 10, 20]
) -> pd.DataFrame:
    """
    Price impact features.

    Args:
        df: DataFrame with OHLCV data
        windows: Windows for calculation

    Returns:
        DataFrame with price impact features
    """
    features = pd.DataFrame(index=df.index)

    returns = df['Close'].pct_change().abs()
    dollar_volume = df['Close'] * df['Volume']

    for window in windows:
        # Kyle's lambda
        features[f'kyle_lambda_{window}'] = kyle_lambda(df, window=window)

        # Price impact per dollar
        features[f'impact_per_dollar_{window}'] = (
            returns.rolling(window).mean() /
            (dollar_volume.rolling(window).mean() + 1e-10)
        )

        # Price impact per volume
        features[f'impact_per_volume_{window}'] = (
            returns.rolling(window).mean() /
            (df['Volume'].rolling(window).mean() + 1e-10)
        )

    return features


def get_market_depth_features(
    df: pd.DataFrame,
    windows: list[int] = [10, 20, 50]
) -> pd.DataFrame:
    """
    Market depth proxy features.

    Args:
        df: DataFrame with OHLCV data
        windows: Windows for calculation

    Returns:
        DataFrame with depth features
    """
    features = pd.DataFrame(index=df.index)

    # Volume at price levels (proxy)
    for window in windows:
        # Average volume
        features[f'avg_volume_{window}'] = df['Volume'].rolling(window).mean()

        # Volume concentration
        total_vol = df['Volume'].rolling(window).sum()
        max_vol = df['Volume'].rolling(window).max()
        features[f'volume_concentration_{window}'] = max_vol / (total_vol + 1e-10)

        # Depth ratio (volume / price range)
        price_range = (df['High'] - df['Low']).rolling(window).mean()
        features[f'depth_ratio_{window}'] = (
            features[f'avg_volume_{window}'] / (price_range + 1e-10)
        )

    return features


def get_tick_features(
    df: pd.DataFrame,
    windows: list[int] = [10, 20, 50]
) -> pd.DataFrame:
    """
    Tick-based features.

    Args:
        df: DataFrame with OHLCV data
        windows: Windows for calculation

    Returns:
        DataFrame with tick features
    """
    features = pd.DataFrame(index=df.index)

    # Tick direction
    price_changes = df['Close'].diff()
    tick_direction = pd.Series(0, index=df.index)
    tick_direction[price_changes > 0] = 1
    tick_direction[price_changes < 0] = -1

    for window in windows:
        # Tick imbalance
        features[f'tick_imbalance_{window}'] = tick_direction.rolling(window).sum()

        # Up-tick percentage
        features[f'uptick_pct_{window}'] = (
            (tick_direction == 1).rolling(window).sum() / window
        )

        # Tick volatility
        features[f'tick_vol_{window}'] = tick_direction.rolling(window).std()

    return features
