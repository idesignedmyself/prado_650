"""
Sample weighting methods.
Implements sequential bootstrap and return-based weighting.
"""
import numpy as np
import pandas as pd
from typing import Optional


def get_sample_weights_by_return(
    returns: pd.Series,
    clip_range: Optional[tuple] = None
) -> pd.Series:
    """
    Weight samples by absolute return.

    Args:
        returns: Return series
        clip_range: Optional (min, max) to clip weights

    Returns:
        Series of sample weights
    """
    weights = abs(returns)

    # Normalize
    weights = weights / weights.sum()

    # Clip if specified
    if clip_range is not None:
        weights = weights.clip(clip_range[0], clip_range[1])
        weights = weights / weights.sum()  # Re-normalize

    return weights


def get_sample_weights_by_time_decay(
    index: pd.Index,
    half_life: int = 100
) -> pd.Series:
    """
    Weight samples by time decay (more recent = higher weight).

    Args:
        index: Time index
        half_life: Half-life for exponential decay

    Returns:
        Series of sample weights
    """
    # Create time-based weights
    n = len(index)
    time_idx = np.arange(n)

    # Exponential decay
    decay = np.exp(-np.log(2) / half_life * (n - 1 - time_idx))

    weights = pd.Series(decay, index=index)
    weights = weights / weights.sum()

    return weights


def get_time_decay_weights(
    t: pd.Series,
    half_life: float = 1.0
) -> pd.Series:
    """
    Calculate time decay weights for each sample.

    Args:
        t: Series of timestamps or indices
        half_life: Half-life for decay

    Returns:
        Series of weights
    """
    if isinstance(t, pd.DatetimeIndex):
        # Convert to numeric
        t_numeric = (t - t[0]).total_seconds()
        t_numeric = pd.Series(t_numeric, index=t)
    else:
        t_numeric = pd.Series(range(len(t)), index=t)

    # Calculate decay
    max_t = t_numeric.max()
    decay = np.exp(-np.log(2) * (max_t - t_numeric) / half_life)

    weights = decay / decay.sum()

    return weights


def get_sequential_bootstrap_weights(
    ind_matrix: pd.DataFrame,
    sample_length: Optional[pd.Series] = None
) -> pd.Series:
    """
    Calculate sample weights using sequential bootstrap.

    Sequential bootstrap accounts for overlapping samples.

    Args:
        ind_matrix: Indicator matrix (samples x time)
        sample_length: Optional series of sample lengths

    Returns:
        Series of sample weights
    """
    if sample_length is None:
        # Calculate average uniqueness
        avg_uniqueness = ind_matrix.mean(axis=1)
    else:
        # Weight by sample length
        avg_uniqueness = ind_matrix.mean(axis=1) / sample_length

    # Normalize
    weights = avg_uniqueness / avg_uniqueness.sum()

    return weights


def get_indicator_matrix(
    triple_barrier_events: pd.DataFrame,
    close: pd.Series
) -> pd.DataFrame:
    """
    Create indicator matrix showing when each sample is active.

    Args:
        triple_barrier_events: Events with t1 (end time)
        close: Price series for time index

    Returns:
        DataFrame indicator matrix (samples x time)
    """
    # Initialize matrix
    ind_matrix = pd.DataFrame(0, index=triple_barrier_events.index, columns=close.index)

    # Fill in active periods
    for idx, row in triple_barrier_events.iterrows():
        if 't1' in row and pd.notna(row['t1']):
            # Get time range
            start = idx
            end = row['t1']

            # Get column indices
            cols = close.loc[start:end].index

            # Set to 1
            ind_matrix.loc[idx, cols] = 1

    return ind_matrix


def get_sample_weights(
    triple_barrier_events: pd.DataFrame,
    close: pd.Series,
    method: str = 'return'
) -> pd.Series:
    """
    Calculate sample weights using specified method.

    Args:
        triple_barrier_events: Labeled events
        close: Price series
        method: Weighting method ('return', 'time_decay', 'sequential')

    Returns:
        Series of sample weights
    """
    if method == 'return':
        if 'ret' in triple_barrier_events.columns:
            return get_sample_weights_by_return(triple_barrier_events['ret'])
        else:
            raise ValueError("Events must have 'ret' column for return weighting")

    elif method == 'time_decay':
        return get_sample_weights_by_time_decay(triple_barrier_events.index)

    elif method == 'sequential':
        ind_matrix = get_indicator_matrix(triple_barrier_events, close)
        return get_sequential_bootstrap_weights(ind_matrix)

    else:
        raise ValueError(f"Unknown weighting method: {method}")


def get_weighted_labels(
    labels: pd.Series,
    weights: pd.Series,
    n_samples: int = 1000
) -> pd.Series:
    """
    Resample labels according to weights.

    Args:
        labels: Label series
        weights: Weight series
        n_samples: Number of samples to draw

    Returns:
        Resampled labels
    """
    # Align
    idx = labels.index.intersection(weights.index)
    labels = labels.loc[idx]
    weights = weights.loc[idx]

    # Normalize weights
    weights = weights / weights.sum()

    # Sample
    sampled_idx = np.random.choice(
        labels.index,
        size=n_samples,
        replace=True,
        p=weights
    )

    return labels.loc[sampled_idx]


def balance_sample_weights(
    labels: pd.Series,
    weights: pd.Series,
    method: str = 'inverse'
) -> pd.Series:
    """
    Balance weights across classes.

    Args:
        labels: Label series
        weights: Current weights
        method: Balancing method ('inverse', 'sqrt_inverse')

    Returns:
        Balanced weights
    """
    # Align
    idx = labels.index.intersection(weights.index)
    labels = labels.loc[idx]
    weights = weights.loc[idx]

    # Calculate class frequencies
    class_counts = labels.value_counts()

    # Calculate class weights
    if method == 'inverse':
        class_weights = 1.0 / class_counts
    elif method == 'sqrt_inverse':
        class_weights = 1.0 / np.sqrt(class_counts)
    else:
        raise ValueError(f"Unknown balancing method: {method}")

    # Apply class weights
    balanced_weights = weights.copy()
    for label, weight in class_weights.items():
        balanced_weights[labels == label] *= weight

    # Normalize
    balanced_weights = balanced_weights / balanced_weights.sum()

    return balanced_weights
