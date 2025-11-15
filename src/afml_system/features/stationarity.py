"""
Stationarity features: fractional differentiation and log returns.
Implements methods from Advances in Financial Machine Learning.
"""
import numpy as np
import pandas as pd
from typing import Optional


def frac_diff(
    series: pd.Series,
    d: float,
    threshold: float = 1e-5
) -> pd.Series:
    """
    Fractional differentiation (FFD).

    Applies fractional differentiation to make series stationary while
    preserving memory.

    Args:
        series: Input series
        d: Differentiation order (0 < d < 1)
        threshold: Threshold for weight truncation

    Returns:
        Fractionally differentiated series
    """
    # Calculate weights
    weights = _get_frac_diff_weights(d, threshold)
    width = len(weights) - 1

    # Apply weights
    output = pd.Series(index=series.index, dtype=float)

    for i in range(width, len(series)):
        if not np.isnan(series.iloc[i]):
            output.iloc[i] = np.dot(weights.T, series.iloc[i-width:i+1])

    return output


def _get_frac_diff_weights(d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Calculate weights for fractional differentiation.

    Args:
        d: Differentiation order
        threshold: Threshold for truncation

    Returns:
        Array of weights
    """
    weights = [1.0]
    k = 1

    while True:
        weight = -weights[-1] * (d - k + 1) / k
        if abs(weight) < threshold:
            break
        weights.append(weight)
        k += 1

    weights = np.array(weights[::-1])
    return weights


def frac_diff_ffd(
    series: pd.Series,
    d: float,
    threshold: float = 1e-5
) -> pd.Series:
    """
    Fixed-width window fractional differentiation.

    More efficient version that uses fixed window.

    Args:
        series: Input series
        d: Differentiation order
        threshold: Threshold for weight truncation

    Returns:
        Fractionally differentiated series
    """
    weights = _get_frac_diff_weights(d, threshold)
    width = len(weights) - 1

    df = {}
    for name in series.index[width:]:
        loc0 = series.index.get_loc(name)
        if not np.isnan(series.iloc[loc0]):
            df[name] = np.dot(weights.T, series.iloc[loc0-width:loc0+1])

    return pd.Series(df)


def log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate log returns.

    Args:
        prices: Price series

    Returns:
        Log returns
    """
    return np.log(prices / prices.shift(1))


def simple_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate simple returns.

    Args:
        prices: Price series

    Returns:
        Simple returns
    """
    return prices.pct_change()


def get_stationary_series(
    series: pd.Series,
    method: str = 'frac_diff',
    d: float = 0.5
) -> pd.Series:
    """
    Get stationary version of series.

    Args:
        series: Input series
        method: Method ('frac_diff', 'log_returns', 'diff', 'pct_change')
        d: Differentiation order (for frac_diff)

    Returns:
        Stationary series
    """
    if method == 'frac_diff':
        return frac_diff(series, d)
    elif method == 'log_returns':
        return log_returns(series)
    elif method == 'diff':
        return series.diff()
    elif method == 'pct_change':
        return series.pct_change()
    else:
        raise ValueError(f"Unknown method: {method}")


def adf_test(series: pd.Series) -> dict:
    """
    Augmented Dickey-Fuller test for stationarity.

    Args:
        series: Series to test

    Returns:
        Dictionary with test results
    """
    try:
        from statsmodels.tsa.stattools import adfuller

        # Remove NaN
        series_clean = series.dropna()

        # Run test
        result = adfuller(series_clean, autolag='AIC')

        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'lags_used': result[2],
            'n_obs': result[3],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    except ImportError:
        # Fallback if statsmodels not available
        return {
            'error': 'statsmodels not available',
            'is_stationary': None
        }


def find_optimal_d(
    series: pd.Series,
    d_range: tuple = (0.0, 1.0),
    step: float = 0.1,
    target_p_value: float = 0.05
) -> float:
    """
    Find optimal differentiation order for stationarity.

    Args:
        series: Input series
        d_range: Range of d values to test
        step: Step size for d
        target_p_value: Target p-value for ADF test

    Returns:
        Optimal d value
    """
    d_values = np.arange(d_range[0], d_range[1] + step, step)

    results = []
    for d in d_values:
        # Apply fractional diff
        diff_series = frac_diff(series, d)

        # Test stationarity
        adf_result = adf_test(diff_series)

        results.append({
            'd': d,
            'p_value': adf_result.get('p_value', 1.0),
            'is_stationary': adf_result.get('is_stationary', False)
        })

    # Find minimum d that achieves stationarity
    for result in results:
        if result['is_stationary']:
            return result['d']

    # If none stationary, return d with lowest p-value
    results_sorted = sorted(results, key=lambda x: x['p_value'])
    return results_sorted[0]['d']


def cumsum_returns(returns: pd.Series) -> pd.Series:
    """
    Calculate cumulative returns.

    Args:
        returns: Return series

    Returns:
        Cumulative return series
    """
    return (1 + returns).cumprod() - 1


def expanding_mean(series: pd.Series) -> pd.Series:
    """
    Calculate expanding mean.

    Args:
        series: Input series

    Returns:
        Expanding mean
    """
    return series.expanding().mean()


def expanding_std(series: pd.Series) -> pd.Series:
    """
    Calculate expanding standard deviation.

    Args:
        series: Input series

    Returns:
        Expanding std
    """
    return series.expanding().std()
