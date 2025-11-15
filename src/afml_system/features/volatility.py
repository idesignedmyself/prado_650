"""
Volatility estimators.
Implements 6 volatility estimation methods.
"""
import numpy as np
import pandas as pd
from typing import Optional


def parkinson_volatility(
    df: pd.DataFrame,
    window: int = 20,
    high_col: str = 'High',
    low_col: str = 'Low',
    trading_periods: int = 252
) -> pd.Series:
    """
    Parkinson volatility estimator.

    Uses high-low range, more efficient than close-to-close.

    Args:
        df: DataFrame with OHLC data
        window: Rolling window
        high_col: High price column
        low_col: Low price column
        trading_periods: Trading periods per year

    Returns:
        Parkinson volatility estimate
    """
    high_low_ratio = np.log(df[high_col] / df[low_col])
    parkinson = high_low_ratio ** 2 / (4 * np.log(2))

    vol = np.sqrt(parkinson.rolling(window).mean() * trading_periods)

    return vol


def garman_klass_volatility(
    df: pd.DataFrame,
    window: int = 20,
    trading_periods: int = 252
) -> pd.Series:
    """
    Garman-Klass volatility estimator.

    More efficient than Parkinson, uses OHLC.

    Args:
        df: DataFrame with OHLC data
        window: Rolling window
        trading_periods: Trading periods per year

    Returns:
        Garman-Klass volatility estimate
    """
    log_hl = np.log(df['High'] / df['Low'])
    log_co = np.log(df['Close'] / df['Open'])

    gk = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2

    vol = np.sqrt(gk.rolling(window).mean() * trading_periods)

    return vol


def rogers_satchell_volatility(
    df: pd.DataFrame,
    window: int = 20,
    trading_periods: int = 252
) -> pd.Series:
    """
    Rogers-Satchell volatility estimator.

    Allows for drift, uses OHLC.

    Args:
        df: DataFrame with OHLC data
        window: Rolling window
        trading_periods: Trading periods per year

    Returns:
        Rogers-Satchell volatility estimate
    """
    log_ho = np.log(df['High'] / df['Open'])
    log_hc = np.log(df['High'] / df['Close'])
    log_lo = np.log(df['Low'] / df['Open'])
    log_lc = np.log(df['Low'] / df['Close'])

    rs = log_ho * log_hc + log_lo * log_lc

    vol = np.sqrt(rs.rolling(window).mean() * trading_periods)

    return vol


def yang_zhang_volatility(
    df: pd.DataFrame,
    window: int = 20,
    trading_periods: int = 252
) -> pd.Series:
    """
    Yang-Zhang volatility estimator.

    Combines overnight and intraday volatility, handles both drift and opening jumps.

    Args:
        df: DataFrame with OHLC data
        window: Rolling window
        trading_periods: Trading periods per year

    Returns:
        Yang-Zhang volatility estimate
    """
    log_ho = np.log(df['High'] / df['Open'])
    log_lo = np.log(df['Low'] / df['Open'])
    log_co = np.log(df['Close'] / df['Open'])

    log_oc = np.log(df['Open'] / df['Close'].shift(1))
    log_cc = np.log(df['Close'] / df['Close'].shift(1))

    # Overnight volatility
    overnight_vol = log_oc.rolling(window).var()

    # Open-to-close volatility (Rogers-Satchell)
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    rs_vol = rs.rolling(window).mean()

    # Close-to-close volatility
    close_vol = log_cc.rolling(window).var()

    # Yang-Zhang estimator
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yz = overnight_vol + k * close_vol + (1 - k) * rs_vol

    vol = np.sqrt(yz * trading_periods)

    return vol


def ewma_volatility(
    returns: pd.Series,
    span: int = 20,
    trading_periods: int = 252
) -> pd.Series:
    """
    Exponentially Weighted Moving Average volatility.

    Args:
        returns: Return series
        span: EWMA span
        trading_periods: Trading periods per year

    Returns:
        EWMA volatility estimate
    """
    vol = returns.ewm(span=span).std() * np.sqrt(trading_periods)
    return vol


def realized_volatility(
    returns: pd.Series,
    window: int = 20,
    trading_periods: int = 252
) -> pd.Series:
    """
    Realized volatility (standard deviation of returns).

    Args:
        returns: Return series
        window: Rolling window
        trading_periods: Trading periods per year

    Returns:
        Realized volatility
    """
    vol = returns.rolling(window).std() * np.sqrt(trading_periods)
    return vol


def get_all_volatility_estimates(
    df: pd.DataFrame,
    window: int = 20,
    trading_periods: int = 252
) -> pd.DataFrame:
    """
    Calculate all volatility estimates.

    Args:
        df: DataFrame with OHLCV data
        window: Rolling window
        trading_periods: Trading periods per year

    Returns:
        DataFrame with all volatility estimates
    """
    vol_df = pd.DataFrame(index=df.index)

    # Calculate returns
    returns = df['Close'].pct_change()

    # All estimators
    vol_df['parkinson'] = parkinson_volatility(df, window, trading_periods=trading_periods)
    vol_df['garman_klass'] = garman_klass_volatility(df, window, trading_periods=trading_periods)
    vol_df['rogers_satchell'] = rogers_satchell_volatility(df, window, trading_periods=trading_periods)
    vol_df['yang_zhang'] = yang_zhang_volatility(df, window, trading_periods=trading_periods)
    vol_df['ewma'] = ewma_volatility(returns, span=window, trading_periods=trading_periods)
    vol_df['realized'] = realized_volatility(returns, window, trading_periods=trading_periods)

    return vol_df


def volatility_cone(
    df: pd.DataFrame,
    windows: list[int] = [5, 10, 20, 50, 100],
    quantiles: list[float] = [0.25, 0.50, 0.75]
) -> pd.DataFrame:
    """
    Calculate volatility cone for different windows.

    Args:
        df: DataFrame with price data
        windows: List of windows to calculate
        quantiles: Quantiles to compute

    Returns:
        DataFrame with volatility cone
    """
    returns = df['Close'].pct_change()

    results = []

    for window in windows:
        vol = returns.rolling(window).std() * np.sqrt(252)

        stats = {
            'window': window,
            'min': vol.min(),
            'max': vol.max(),
            'mean': vol.mean(),
            'current': vol.iloc[-1]
        }

        for q in quantiles:
            stats[f'q{int(q*100)}'] = vol.quantile(q)

        results.append(stats)

    return pd.DataFrame(results)


def garch_volatility(
    returns: pd.Series,
    p: int = 1,
    q: int = 1
) -> Optional[pd.Series]:
    """
    GARCH volatility forecast.

    Args:
        returns: Return series
        p: GARCH p parameter
        q: GARCH q parameter

    Returns:
        GARCH volatility series or None if arch not available
    """
    try:
        from arch import arch_model

        # Fit GARCH model
        model = arch_model(returns.dropna() * 100, vol='Garch', p=p, q=q)
        fitted = model.fit(disp='off')

        # Get conditional volatility
        vol = fitted.conditional_volatility / 100

        # Align with original index
        vol_series = pd.Series(index=returns.index, dtype=float)
        vol_series.loc[vol.index] = vol.values

        return vol_series

    except ImportError:
        return None
