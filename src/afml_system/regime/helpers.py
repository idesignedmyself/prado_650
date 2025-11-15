"""
Helper functions for regime detection.
Provides utility functions like ADX, EMA slope, volume z-score.
"""
import numpy as np
import pandas as pd


def calculate_adx(
    df: pd.DataFrame,
    period: int = 14
) -> pd.Series:
    """
    Calculate Average Directional Index (ADX).

    Args:
        df: DataFrame with High, Low, Close
        period: ADX period

    Returns:
        ADX values
    """
    high = df['High']
    low = df['Low']
    close = df['Close']

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift()
    down_move = low.shift() - low

    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)

    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

    # Smoothed ATR
    atr = tr.rolling(period).mean()

    # Directional Indicators
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(period).mean()

    return adx


def calculate_ema_slope(
    series: pd.Series,
    period: int = 50,
    lookback: int = 5
) -> pd.Series:
    """
    Calculate slope of EMA.

    Args:
        series: Price series
        period: EMA period
        lookback: Lookback for slope calculation

    Returns:
        EMA slope values
    """
    ema_vals = series.ewm(span=period, adjust=False).mean()

    # Calculate slope
    slope = (ema_vals - ema_vals.shift(lookback)) / lookback

    return slope


def calculate_volume_zscore(
    volume: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Calculate volume z-score.

    Args:
        volume: Volume series
        window: Rolling window for mean/std

    Returns:
        Volume z-score
    """
    vol_mean = volume.rolling(window).mean()
    vol_std = volume.rolling(window).std()

    zscore = (volume - vol_mean) / (vol_std + 1e-10)

    return zscore


def calculate_volatility_regime(
    returns: pd.Series,
    window: int = 20,
    threshold: float = 1.0
) -> pd.Series:
    """
    Calculate volatility regime (high/low).

    Args:
        returns: Return series
        window: Window for volatility
        threshold: Z-score threshold

    Returns:
        Series with regime: 1 = high vol, 0 = normal, -1 = low vol
    """
    vol = returns.rolling(window).std()
    vol_mean = vol.rolling(window * 2).mean()
    vol_std = vol.rolling(window * 2).std()

    vol_zscore = (vol - vol_mean) / (vol_std + 1e-10)

    regime = pd.Series(0, index=returns.index)
    regime[vol_zscore > threshold] = 1
    regime[vol_zscore < -threshold] = -1

    return regime


def calculate_trend_strength(
    df: pd.DataFrame,
    period: int = 14
) -> pd.Series:
    """
    Calculate trend strength using ADX.

    Args:
        df: DataFrame with OHLC
        period: Calculation period

    Returns:
        Trend strength (0-100)
    """
    return calculate_adx(df, period)


def calculate_price_acceleration(
    prices: pd.Series,
    window: int = 5
) -> pd.Series:
    """
    Calculate price acceleration (rate of change of returns).

    Args:
        prices: Price series
        window: Window for calculation

    Returns:
        Price acceleration
    """
    returns = prices.pct_change()
    acceleration = returns.diff(window)

    return acceleration


def calculate_regime_change_probability(
    regime_history: pd.Series,
    window: int = 50
) -> pd.Series:
    """
    Calculate probability of regime change.

    Args:
        regime_history: Historical regime labels
        window: Window for probability estimation

    Returns:
        Regime change probability
    """
    regime_changes = (regime_history != regime_history.shift()).astype(int)
    change_prob = regime_changes.rolling(window).mean()

    return change_prob


def smooth_regime(
    regime: pd.Series,
    window: int = 3
) -> pd.Series:
    """
    Smooth regime labels to reduce noise.

    Args:
        regime: Regime labels
        window: Smoothing window

    Returns:
        Smoothed regime labels
    """
    # Mode smoothing
    smoothed = regime.rolling(window, center=True).apply(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        raw=False
    )

    return smoothed


def calculate_market_breadth(
    returns_matrix: pd.DataFrame,
    threshold: float = 0.0
) -> pd.Series:
    """
    Calculate market breadth (percentage of positive returns).

    Args:
        returns_matrix: DataFrame with returns for multiple assets
        threshold: Threshold for positive return

    Returns:
        Market breadth series
    """
    positive_returns = (returns_matrix > threshold).sum(axis=1)
    breadth = positive_returns / len(returns_matrix.columns)

    return breadth


def calculate_regime_persistence(
    regime: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Calculate how persistent the current regime is.

    Args:
        regime: Regime labels
        window: Window for persistence calculation

    Returns:
        Persistence score (0-1)
    """
    current_regime = regime.copy()
    persistence = pd.Series(index=regime.index, dtype=float)

    for i in range(len(regime)):
        if i < window:
            persistence.iloc[i] = np.nan
        else:
            window_regimes = regime.iloc[i-window:i]
            current = regime.iloc[i]
            persistence.iloc[i] = (window_regimes == current).sum() / window

    return persistence


def get_regime_stats(
    regime: pd.Series,
    returns: pd.Series
) -> pd.DataFrame:
    """
    Calculate statistics for each regime.

    Args:
        regime: Regime labels
        returns: Return series

    Returns:
        DataFrame with regime statistics
    """
    stats = []

    for regime_label in regime.unique():
        regime_mask = regime == regime_label
        regime_returns = returns[regime_mask]

        stats.append({
            'regime': regime_label,
            'count': regime_mask.sum(),
            'frequency': regime_mask.sum() / len(regime),
            'mean_return': regime_returns.mean(),
            'std_return': regime_returns.std(),
            'sharpe': regime_returns.mean() / (regime_returns.std() + 1e-10) * np.sqrt(252),
            'max_return': regime_returns.max(),
            'min_return': regime_returns.min()
        })

    return pd.DataFrame(stats)
