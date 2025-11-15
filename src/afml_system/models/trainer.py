"""
Model training functions.
Implements training for primary, meta, and strategy models.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from .purged_kfold import PurgedKFold, cv_score


def train_primary_model(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[pd.Series] = None,
    samples_info_sets: Optional[pd.Series] = None,
    model_type: str = 'rf',
    model_params: Optional[Dict] = None,
    cv_folds: int = 5
) -> Tuple[Any, Dict]:
    """
    Train primary prediction model.

    Args:
        X: Feature matrix
        y: Labels
        sample_weight: Sample weights
        samples_info_sets: Sample end times for purging
        model_type: Model type ('rf', 'gb')
        model_params: Model parameters
        cv_folds: Number of CV folds

    Returns:
        Tuple of (trained_model, metrics)
    """
    # Default parameters
    if model_params is None:
        if model_type == 'rf':
            model_params = {
                'n_estimators': 100,
                'max_depth': 5,
                'min_samples_leaf': 50,
                'random_state': 42,
                'n_jobs': -1
            }
        elif model_type == 'gb':
            model_params = {
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.1,
                'random_state': 42
            }

    # Create model
    if model_type == 'rf':
        model = RandomForestClassifier(**model_params)
    elif model_type == 'gb':
        model = GradientBoostingClassifier(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Cross-validation
    if samples_info_sets is not None:
        cv_gen = PurgedKFold(n_splits=cv_folds, samples_info_sets=samples_info_sets)
    else:
        cv_gen = PurgedKFold(n_splits=cv_folds)

    cv_results = cv_score(
        model,
        X,
        y,
        sample_weight=sample_weight,
        cv_gen=cv_gen,
        scoring='accuracy'
    )

    # Train final model on all data
    if sample_weight is not None:
        model.fit(X, y, sample_weight=sample_weight)
    else:
        model.fit(X, y)

    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
    else:
        feature_importance = None

    # Compile metrics
    metrics = {
        'cv_scores': cv_results['scores'],
        'cv_mean': cv_results['mean_score'],
        'cv_std': cv_results['std_score'],
        'n_samples': len(X),
        'n_features': X.shape[1],
        'feature_importance': feature_importance
    }

    return model, metrics


def train_meta_model(
    X_meta: pd.DataFrame,
    y_meta: pd.Series,
    sample_weight: Optional[pd.Series] = None,
    model_params: Optional[Dict] = None
) -> Tuple[Any, Dict]:
    """
    Train meta-labeling model.

    Args:
        X_meta: Meta-features (including primary predictions)
        y_meta: Meta-labels (whether to take position)
        sample_weight: Sample weights
        model_params: Model parameters

    Returns:
        Tuple of (trained_model, metrics)
    """
    # Default parameters for meta-model (simpler than primary)
    if model_params is None:
        model_params = {
            'n_estimators': 50,
            'max_depth': 3,
            'min_samples_leaf': 100,
            'random_state': 42,
            'n_jobs': -1
        }

    model = RandomForestClassifier(**model_params)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_meta, y_meta, test_size=0.2, random_state=42
    )

    # Get sample weights
    if sample_weight is not None:
        sw_train = sample_weight.loc[X_train.index]
        sw_test = sample_weight.loc[X_test.index]
    else:
        sw_train = None
        sw_test = None

    # Train
    if sw_train is not None:
        model.fit(X_train, y_train, sample_weight=sw_train)
    else:
        model.fit(X_train, y_train)

    # Evaluate
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_pred = model.predict(X_test)

    metrics = {
        'train_size': len(X_train),
        'test_size': len(X_test),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }

    return model, metrics


def train_strategy_models(
    features_dict: Dict[str, pd.DataFrame],
    labels_dict: Dict[str, pd.Series],
    model_type: str = 'rf',
    model_params: Optional[Dict] = None
) -> Dict[str, Tuple[Any, Dict]]:
    """
    Train separate models for each strategy.

    Args:
        features_dict: Dictionary of strategy_name -> features
        labels_dict: Dictionary of strategy_name -> labels
        model_type: Model type
        model_params: Model parameters

    Returns:
        Dictionary of strategy_name -> (model, metrics)
    """
    strategy_models = {}

    for strategy_name in features_dict.keys():
        if strategy_name not in labels_dict:
            continue

        X = features_dict[strategy_name]
        y = labels_dict[strategy_name]

        # Align X and y
        idx = X.index.intersection(y.index)
        X = X.loc[idx]
        y = y.loc[idx]

        if len(X) < 100:
            print(f"Insufficient data for {strategy_name}: {len(X)} samples")
            continue

        # Train model
        model, metrics = train_primary_model(
            X, y,
            model_type=model_type,
            model_params=model_params,
            cv_folds=3
        )

        strategy_models[strategy_name] = (model, metrics)

        print(f"Trained {strategy_name}: CV score = {metrics['cv_mean']:.4f} +/- {metrics['cv_std']:.4f}")

    return strategy_models


def train_regime_specific_models(
    X: pd.DataFrame,
    y: pd.Series,
    regime: pd.Series,
    model_params: Optional[Dict] = None
) -> Dict[str, Tuple[Any, Dict]]:
    """
    Train separate models for each regime.

    Args:
        X: Features
        y: Labels
        regime: Regime labels
        model_params: Model parameters

    Returns:
        Dictionary of regime -> (model, metrics)
    """
    regime_models = {}

    # Align data
    idx = X.index.intersection(y.index).intersection(regime.index)
    X = X.loc[idx]
    y = y.loc[idx]
    regime = regime.loc[idx]

    for regime_label in regime.unique():
        # Filter by regime
        regime_mask = regime == regime_label
        X_regime = X[regime_mask]
        y_regime = y[regime_mask]

        if len(X_regime) < 100:
            print(f"Insufficient data for regime {regime_label}: {len(X_regime)} samples")
            continue

        # Train model
        model, metrics = train_primary_model(
            X_regime,
            y_regime,
            model_type='rf',
            model_params=model_params,
            cv_folds=3
        )

        regime_models[regime_label] = (model, metrics)

        print(f"Trained regime {regime_label}: CV score = {metrics['cv_mean']:.4f}")

    return regime_models


def update_model_online(
    model: Any,
    X_new: pd.DataFrame,
    y_new: pd.Series,
    sample_weight: Optional[pd.Series] = None
) -> Any:
    """
    Update model with new data (partial fit if supported).

    Args:
        model: Existing model
        X_new: New features
        y_new: New labels
        sample_weight: Sample weights

    Returns:
        Updated model
    """
    if hasattr(model, 'partial_fit'):
        # Online learning supported
        if sample_weight is not None:
            model.partial_fit(X_new, y_new, sample_weight=sample_weight)
        else:
            model.partial_fit(X_new, y_new)
    else:
        # Retrain on all data (if feasible)
        if sample_weight is not None:
            model.fit(X_new, y_new, sample_weight=sample_weight)
        else:
            model.fit(X_new, y_new)

    return model
