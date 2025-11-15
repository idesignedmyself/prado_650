"""
Performance evaluation metrics.
Implements sharpe, sortino, calmar, max_drawdown, win_rate.
"""
import numpy as np
import pandas as pd
from typing import Optional


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Trading periods per year

    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    mean_return = excess_returns.mean()
    std_return = excess_returns.std()

    if std_return == 0:
        return 0.0

    sharpe = mean_return / std_return * np.sqrt(periods_per_year)

    return sharpe


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio (uses downside deviation).

    Args:
        returns: Return series
        risk_free_rate: Risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    mean_return = excess_returns.mean()

    # Downside deviation
    negative_returns = excess_returns[excess_returns < 0]
    if len(negative_returns) == 0:
        return np.inf if mean_return > 0 else 0.0

    downside_std = np.sqrt((negative_returns ** 2).mean())

    if downside_std == 0:
        return 0.0

    sortino = mean_return / downside_std * np.sqrt(periods_per_year)

    return sortino


def max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown.

    Args:
        returns: Return series

    Returns:
        Maximum drawdown (positive number)
    """
    if len(returns) < 2:
        return 0.0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max

    max_dd = abs(drawdown.min())

    return max_dd


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (return / max drawdown).

    Args:
        returns: Return series
        periods_per_year: Trading periods per year

    Returns:
        Calmar ratio
    """
    if len(returns) < 2:
        return 0.0

    annual_return = returns.mean() * periods_per_year
    max_dd = max_drawdown(returns)

    if max_dd == 0:
        return 0.0

    calmar = annual_return / max_dd

    return calmar


def win_rate(returns: pd.Series) -> float:
    """
    Calculate win rate (percentage of profitable periods).

    Args:
        returns: Return series

    Returns:
        Win rate (0 to 1)
    """
    if len(returns) == 0:
        return 0.0

    winning_periods = (returns > 0).sum()
    total_periods = len(returns)

    return winning_periods / total_periods


def profit_factor(returns: pd.Series) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    Args:
        returns: Return series

    Returns:
        Profit factor
    """
    if len(returns) == 0:
        return 0.0

    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())

    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def get_all_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> dict:
    """
    Calculate all performance metrics.

    Args:
        returns: Return series
        risk_free_rate: Risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Dictionary of metrics
    """
    if len(returns) < 2:
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'annual_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'num_periods': len(returns)
        }

    total_return = (1 + returns).prod() - 1
    annual_return = returns.mean() * periods_per_year
    annual_vol = returns.std() * np.sqrt(periods_per_year)

    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe_ratio(returns, risk_free_rate, periods_per_year),
        'sortino_ratio': sortino_ratio(returns, risk_free_rate, periods_per_year),
        'calmar_ratio': calmar_ratio(returns, periods_per_year),
        'max_drawdown': max_drawdown(returns),
        'win_rate': win_rate(returns),
        'profit_factor': profit_factor(returns),
        'num_periods': len(returns),
        'avg_return': returns.mean(),
        'median_return': returns.median(),
        'std_return': returns.std(),
        'skewness': returns.skew(),
        'kurtosis': returns.kurtosis()
    }

    return metrics


def calculate_drawdown_series(returns: pd.Series) -> pd.Series:
    """
    Calculate drawdown series.

    Args:
        returns: Return series

    Returns:
        Drawdown series
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max

    return drawdown


def recovery_time(returns: pd.Series) -> Optional[int]:
    """
    Calculate time to recover from max drawdown.

    Args:
        returns: Return series

    Returns:
        Number of periods to recovery (None if not recovered)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max

    max_dd_idx = drawdown.idxmin()
    max_dd_loc = returns.index.get_loc(max_dd_idx)

    # Find recovery
    after_dd = cumulative.iloc[max_dd_loc:]
    peak_value = running_max.iloc[max_dd_loc]

    recovery_idx = after_dd[after_dd >= peak_value]

    if len(recovery_idx) == 0:
        return None  # Not recovered

    recovery_loc = returns.index.get_loc(recovery_idx.index[0])
    recovery_periods = recovery_loc - max_dd_loc

    return recovery_periods
