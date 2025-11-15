"""
Meta-labeling for strategy refinement.
Implements meta-labeling from Advances in Financial Machine Learning.
"""
import numpy as np
import pandas as pd
from typing import Optional


def get_meta_labels(
    primary_predictions: pd.Series,
    triple_barrier_events: pd.DataFrame,
    threshold: float = 0.0
) -> pd.DataFrame:
    """
    Create meta-labels for secondary model.

    Meta-labels indicate whether to take the primary model's prediction or not.

    Args:
        primary_predictions: Primary model predictions (1 or -1)
        triple_barrier_events: Events from triple barrier method
        threshold: Threshold for filtering predictions

    Returns:
        DataFrame with meta-labels (1 = take position, 0 = skip)
    """
    # Align indices
    events = triple_barrier_events.copy()
    events = events.loc[events.index.intersection(primary_predictions.index)]

    # Meta-label is 1 if the trade was profitable, 0 otherwise
    meta_labels = pd.DataFrame(index=events.index)
    meta_labels['side'] = primary_predictions.loc[events.index]
    meta_labels['ret'] = events['ret']
    meta_labels['label'] = events['label']

    # Adjust return by side
    meta_labels['sided_ret'] = meta_labels['ret'] * meta_labels['side']

    # Meta-label: 1 if profitable, 0 otherwise
    meta_labels['meta_label'] = (meta_labels['sided_ret'] > threshold).astype(int)

    return meta_labels


def apply_meta_model(
    primary_predictions: pd.Series,
    meta_predictions: pd.Series
) -> pd.Series:
    """
    Apply meta-model predictions to filter primary predictions.

    Args:
        primary_predictions: Primary model predictions (side)
        meta_predictions: Meta-model predictions (confidence)

    Returns:
        Filtered predictions (0 where meta-model says to skip)
    """
    # Align indices
    aligned_meta = meta_predictions.reindex(primary_predictions.index, fill_value=0)

    # Filter: keep primary prediction only where meta-model says yes
    filtered = primary_predictions.copy()
    filtered[aligned_meta == 0] = 0

    return filtered


def get_meta_labels_with_sizing(
    primary_predictions: pd.Series,
    triple_barrier_events: pd.DataFrame,
    return_threshold: float = 0.0
) -> pd.DataFrame:
    """
    Create meta-labels with position sizing information.

    Args:
        primary_predictions: Primary model predictions
        triple_barrier_events: Events from triple barrier
        return_threshold: Threshold for positive label

    Returns:
        DataFrame with meta-labels and sizing info
    """
    events = triple_barrier_events.copy()
    events = events.loc[events.index.intersection(primary_predictions.index)]

    meta_data = pd.DataFrame(index=events.index)
    meta_data['primary_side'] = primary_predictions.loc[events.index]
    meta_data['ret'] = events['ret']
    meta_data['label'] = events['label']

    # Calculate sided return
    meta_data['sided_ret'] = meta_data['ret'] * meta_data['primary_side']

    # Meta-label
    meta_data['meta_label'] = (meta_data['sided_ret'] > return_threshold).astype(int)

    # Position size (absolute return as proxy for confidence)
    meta_data['size'] = abs(meta_data['sided_ret'])

    return meta_data


def meta_label_cv_score(
    y_true: pd.Series,
    y_pred: pd.Series,
    primary_side: pd.Series
) -> dict:
    """
    Calculate meta-labeling specific metrics.

    Args:
        y_true: True meta-labels
        y_pred: Predicted meta-labels
        primary_side: Primary model predictions

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Align
    idx = y_true.index.intersection(y_pred.index).intersection(primary_side.index)
    y_true = y_true.loc[idx]
    y_pred = y_pred.loc[idx]
    primary_side = primary_side.loc[idx]

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Meta-specific: precision of filtered signals
    filtered_signals = primary_side[y_pred == 1]
    true_positives = ((y_true == 1) & (y_pred == 1)).sum()
    false_positives = ((y_true == 0) & (y_pred == 1)).sum()

    signal_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'signal_precision': signal_precision,
        'signals_kept': (y_pred == 1).sum(),
        'signals_total': len(y_pred)
    }


def combine_primary_meta_predictions(
    primary_model,
    meta_model,
    X_primary: pd.DataFrame,
    X_meta: pd.DataFrame
) -> pd.Series:
    """
    Generate final predictions using both models.

    Args:
        primary_model: Trained primary model
        meta_model: Trained meta-model
        X_primary: Features for primary model
        X_meta: Features for meta-model

    Returns:
        Final predictions
    """
    # Get primary predictions
    primary_pred = pd.Series(
        primary_model.predict(X_primary),
        index=X_primary.index
    )

    # Get meta predictions
    meta_pred = pd.Series(
        meta_model.predict(X_meta),
        index=X_meta.index
    )

    # Combine
    final_pred = apply_meta_model(primary_pred, meta_pred)

    return final_pred


def get_meta_features(
    X: pd.DataFrame,
    primary_predictions: pd.Series,
    primary_proba: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Create features for meta-model.

    Args:
        X: Original features
        primary_predictions: Primary model predictions
        primary_proba: Primary model probabilities (optional)

    Returns:
        DataFrame with meta-features
    """
    meta_features = X.copy()

    # Add primary prediction
    meta_features['primary_pred'] = primary_predictions

    # Add probability features if available
    if primary_proba is not None:
        if isinstance(primary_proba, pd.DataFrame):
            for col in primary_proba.columns:
                meta_features[f'primary_proba_{col}'] = primary_proba[col]
        else:
            meta_features['primary_proba'] = primary_proba

        # Confidence (distance from 0.5)
        if 'primary_proba_1' in meta_features.columns:
            meta_features['confidence'] = abs(meta_features['primary_proba_1'] - 0.5)

    return meta_features
