"""
Triple barrier method for labeling.
Implements the triple barrier method from Advances in Financial Machine Learning.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


def get_daily_volatility(close: pd.Series, span: int = 100) -> pd.Series:
    """
    Calculate daily volatility using exponentially weighted moving average.

    Args:
        close: Close price series
        span: EWMA span

    Returns:
        Series of daily volatility
    """
    returns = close.pct_change()
    volatility = returns.ewm(span=span).std()
    return volatility


def triple_barrier_labels(
    close: pd.Series,
    events: pd.DatetimeIndex,
    pt_sl: list[float],
    min_ret: float = 0.0,
    num_threads: int = 1,
    vertical_barrier_days: int = 5,
    side: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Apply triple barrier method to label events.

    Args:
        close: Close price series
        events: Event timestamps to label
        pt_sl: [profit_taking_multiplier, stop_loss_multiplier] for volatility
        min_ret: Minimum return to consider
        num_threads: Number of threads for parallel processing
        vertical_barrier_days: Days until vertical barrier
        side: Optional series of position sides (1 or -1)

    Returns:
        DataFrame with columns: t1 (barrier hit time), ret (return), label (1/0/-1)
    """
    # Get target volatility
    target = get_daily_volatility(close)

    # Create vertical barriers
    vertical_barriers = pd.Series(index=events, dtype='datetime64[ns]')
    for event in events:
        loc = close.index.get_loc(event)
        if loc + vertical_barrier_days < len(close):
            vertical_barriers[event] = close.index[loc + vertical_barrier_days]
        else:
            vertical_barriers[event] = close.index[-1]

    # Apply barriers
    if num_threads > 1:
        results = _apply_barriers_parallel(
            close, events, pt_sl, target, min_ret, vertical_barriers, side, num_threads
        )
    else:
        results = _apply_barriers_sequential(
            close, events, pt_sl, target, min_ret, vertical_barriers, side
        )

    return results


def _apply_barriers_sequential(
    close: pd.Series,
    events: pd.DatetimeIndex,
    pt_sl: list[float],
    target: pd.Series,
    min_ret: float,
    vertical_barriers: pd.Series,
    side: Optional[pd.Series]
) -> pd.DataFrame:
    """Apply barriers sequentially."""
    results = []

    for event in events:
        result = _apply_single_barrier(
            close, event, pt_sl, target, min_ret, vertical_barriers[event], side
        )
        if result is not None:
            results.append(result)

    if results:
        df = pd.DataFrame(results)
        df.set_index('t0', inplace=True)
        return df
    else:
        return pd.DataFrame(columns=['t1', 'ret', 'label'])


def _apply_barriers_parallel(
    close: pd.Series,
    events: pd.DatetimeIndex,
    pt_sl: list[float],
    target: pd.Series,
    min_ret: float,
    vertical_barriers: pd.Series,
    side: Optional[pd.Series],
    num_threads: int
) -> pd.DataFrame:
    """Apply barriers in parallel."""
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        tasks = [
            (close, event, pt_sl, target, min_ret, vertical_barriers[event], side)
            for event in events
        ]
        results = list(executor.map(_apply_single_barrier_wrapper, tasks))

    results = [r for r in results if r is not None]

    if results:
        df = pd.DataFrame(results)
        df.set_index('t0', inplace=True)
        return df
    else:
        return pd.DataFrame(columns=['t1', 'ret', 'label'])


def _apply_single_barrier_wrapper(args):
    """Wrapper for parallel processing."""
    return _apply_single_barrier(*args)


def _apply_single_barrier(
    close: pd.Series,
    event: pd.Timestamp,
    pt_sl: list[float],
    target: pd.Series,
    min_ret: float,
    vertical_barrier: pd.Timestamp,
    side: Optional[pd.Series]
) -> Optional[dict]:
    """
    Apply triple barrier to a single event.

    Returns:
        Dictionary with t0, t1, ret, label or None
    """
    try:
        # Get price at event
        start_price = close.loc[event]

        # Get volatility at event
        if event in target.index:
            vol = target.loc[event]
        else:
            vol = target.iloc[-1]

        # Set barriers
        profit_taking = pt_sl[0] * vol
        stop_loss = pt_sl[1] * vol

        # Get position side
        if side is not None and event in side.index:
            position_side = side.loc[event]
        else:
            position_side = 1

        # Get future prices
        future_prices = close.loc[event:vertical_barrier]

        if len(future_prices) < 2:
            return None

        # Calculate returns
        returns = (future_prices / start_price - 1) * position_side

        # Check barriers
        hit_profit = returns[returns >= profit_taking]
        hit_loss = returns[returns <= -stop_loss]

        # Determine which barrier was hit first
        t1 = vertical_barrier
        ret = returns.iloc[-1]
        label = 0

        if not hit_profit.empty and not hit_loss.empty:
            # Both hit - take earliest
            if hit_profit.index[0] < hit_loss.index[0]:
                t1 = hit_profit.index[0]
                ret = hit_profit.iloc[0]
                label = 1
            else:
                t1 = hit_loss.index[0]
                ret = hit_loss.iloc[0]
                label = -1
        elif not hit_profit.empty:
            t1 = hit_profit.index[0]
            ret = hit_profit.iloc[0]
            label = 1
        elif not hit_loss.empty:
            t1 = hit_loss.index[0]
            ret = hit_loss.iloc[0]
            label = -1
        else:
            # Vertical barrier hit
            if abs(ret) < min_ret:
                label = 0
            else:
                label = 1 if ret > 0 else -1

        return {
            't0': event,
            't1': t1,
            'ret': ret,
            'label': label
        }

    except Exception as e:
        return None


def get_bins(triple_barrier_events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """
    Get labels from triple barrier events.

    Args:
        triple_barrier_events: Output from triple_barrier_labels
        close: Close price series

    Returns:
        DataFrame with labels
    """
    events = triple_barrier_events.copy()

    # Ensure proper columns
    if 'label' not in events.columns:
        events['label'] = 0
        events.loc[events['ret'] > 0, 'label'] = 1
        events.loc[events['ret'] < 0, 'label'] = -1

    return events[['t1', 'label', 'ret']]


def drop_labels(events: pd.DataFrame, min_pct: float = 0.05) -> pd.DataFrame:
    """
    Drop under-represented labels.

    Args:
        events: Labeled events
        min_pct: Minimum percentage threshold

    Returns:
        Filtered events
    """
    # Count labels
    counts = events['label'].value_counts()
    total = len(events)

    # Find labels below threshold
    to_drop = counts[counts / total < min_pct].index

    # Filter
    filtered = events[~events['label'].isin(to_drop)]

    return filtered
