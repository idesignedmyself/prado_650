"""
Technical indicator features.
Standard technical analysis indicators for ML features.
"""
import numpy as np
import pandas as pd
from typing import Optional


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index.

    Args:
        series: Price series
        window: RSI window

    Returns:
        RSI values (0-100)
    """
    delta = series.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence.

    Args:
        series: Price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period

    Returns:
        DataFrame with MACD, signal, and histogram
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)

    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    return pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    })


def bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> pd.DataFrame:
    """
    Bollinger Bands.

    Args:
        series: Price series
        window: Moving average window
        num_std: Number of standard deviations

    Returns:
        DataFrame with upper, middle, lower bands
    """
    middle = sma(series, window)
    std = series.rolling(window).std()

    upper = middle + (std * num_std)
    lower = middle - (std * num_std)

    return pd.DataFrame({
        'bb_upper': upper,
        'bb_middle': middle,
        'bb_lower': lower,
        'bb_width': upper - lower,
        'bb_pct': (series - lower) / (upper - lower + 1e-10)
    })


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average True Range.

    Args:
        df: DataFrame with High, Low, Close
        window: ATR window

    Returns:
        ATR values
    """
    high = df['High']
    low = df['Low']
    close = df['Close']

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.rolling(window).mean()

    return atr_val


def adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Average Directional Index.

    Args:
        df: DataFrame with High, Low, Close
        window: ADX window

    Returns:
        DataFrame with ADX, +DI, -DI
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

    # Smoothed values
    atr_val = tr.rolling(window).mean()
    plus_di = 100 * (plus_dm.rolling(window).mean() / atr_val)
    minus_di = 100 * (minus_dm.rolling(window).mean() / atr_val)

    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_val = dx.rolling(window).mean()

    return pd.DataFrame({
        'adx': adx_val,
        'plus_di': plus_di,
        'minus_di': minus_di
    })


def stochastic(
    df: pd.DataFrame,
    window: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3
) -> pd.DataFrame:
    """
    Stochastic Oscillator.

    Args:
        df: DataFrame with High, Low, Close
        window: Stochastic window
        smooth_k: %K smoothing
        smooth_d: %D smoothing

    Returns:
        DataFrame with %K and %D
    """
    high = df['High']
    low = df['Low']
    close = df['Close']

    lowest_low = low.rolling(window).min()
    highest_high = high.rolling(window).max()

    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    k_smooth = k.rolling(smooth_k).mean()
    d = k_smooth.rolling(smooth_d).mean()

    return pd.DataFrame({
        'stoch_k': k_smooth,
        'stoch_d': d
    })


def obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume.

    Args:
        df: DataFrame with Close and Volume

    Returns:
        OBV values
    """
    direction = pd.Series(0, index=df.index)
    direction[df['Close'] > df['Close'].shift(1)] = 1
    direction[df['Close'] < df['Close'].shift(1)] = -1

    obv_val = (direction * df['Volume']).cumsum()

    return obv_val


def vwap(df: pd.DataFrame, window: Optional[int] = None) -> pd.Series:
    """
    Volume Weighted Average Price.

    Args:
        df: DataFrame with High, Low, Close, Volume
        window: Optional rolling window (None for cumulative)

    Returns:
        VWAP values
    """
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    dollar_volume = typical_price * df['Volume']

    if window is None:
        vwap_val = dollar_volume.cumsum() / df['Volume'].cumsum()
    else:
        vwap_val = dollar_volume.rolling(window).sum() / df['Volume'].rolling(window).sum()

    return vwap_val


def get_technical_features(
    df: pd.DataFrame,
    windows: list[int] = [5, 10, 20, 50]
) -> pd.DataFrame:
    """
    Create comprehensive technical indicator feature set.

    Args:
        df: DataFrame with OHLCV data
        windows: Windows for indicators

    Returns:
        DataFrame with technical features
    """
    features = pd.DataFrame(index=df.index)

    # Moving averages
    for window in windows:
        features[f'sma_{window}'] = sma(df['Close'], window)
        features[f'ema_{window}'] = ema(df['Close'], window)

        # Price vs MA
        features[f'price_sma_ratio_{window}'] = df['Close'] / features[f'sma_{window}']
        features[f'price_ema_ratio_{window}'] = df['Close'] / features[f'ema_{window}']

    # RSI
    for window in [7, 14, 21]:
        features[f'rsi_{window}'] = rsi(df['Close'], window)

    # MACD
    macd_df = macd(df['Close'])
    features['macd'] = macd_df['macd']
    features['macd_signal'] = macd_df['signal']
    features['macd_histogram'] = macd_df['histogram']

    # Bollinger Bands
    bb = bollinger_bands(df['Close'])
    for col in bb.columns:
        features[col] = bb[col]

    # ATR
    for window in [7, 14, 21]:
        features[f'atr_{window}'] = atr(df, window)

    # ADX
    adx_df = adx(df)
    features['adx'] = adx_df['adx']
    features['plus_di'] = adx_df['plus_di']
    features['minus_di'] = adx_df['minus_di']

    # Stochastic
    stoch = stochastic(df)
    features['stoch_k'] = stoch['stoch_k']
    features['stoch_d'] = stoch['stoch_d']

    # OBV
    features['obv'] = obv(df)
    features['obv_ema'] = ema(features['obv'], 20)

    # VWAP
    features['vwap_20'] = vwap(df, 20)
    features['price_vwap_ratio'] = df['Close'] / features['vwap_20']

    # Momentum
    for window in windows:
        features[f'momentum_{window}'] = df['Close'].pct_change(window)
        features[f'roc_{window}'] = (df['Close'] - df['Close'].shift(window)) / df['Close'].shift(window)

    return features
