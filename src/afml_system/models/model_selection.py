"""
Model selection and hyperparameter tuning.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from .purged_kfold import PurgedKFold


def hyperparameter_tuning(
    X: pd.DataFrame,
    y: pd.Series,
    model_class=RandomForestClassifier,
    param_grid: Optional[Dict] = None,
    samples_info_sets: Optional[pd.Series] = None,
    n_iter: int = 20,
    cv_folds: int = 3,
    method: str = 'random'
) -> Tuple[Any, Dict]:
    """
    Hyperparameter tuning with purged CV.

    Args:
        X: Features
        y: Labels
        model_class: Model class to tune
        param_grid: Parameter grid
        samples_info_sets: Sample end times
        n_iter: Number of iterations for random search
        cv_folds: CV folds
        method: 'random' or 'grid'

    Returns:
        Tuple of (best_model, best_params)
    """
    # Default parameter grid for RF
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, 10],
            'min_samples_leaf': [20, 50, 100],
            'max_features': ['sqrt', 'log2', 0.5]
        }

    # Create CV splitter
    if samples_info_sets is not None:
        cv = PurgedKFold(n_splits=cv_folds, samples_info_sets=samples_info_sets)
    else:
        cv = PurgedKFold(n_splits=cv_folds)

    # Create search
    if method == 'random':
        search = RandomizedSearchCV(
            model_class(),
            param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )
    else:
        search = GridSearchCV(
            model_class(),
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )

    # Fit
    search.fit(X, y)

    return search.best_estimator_, search.best_params_


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = 'importance',
    n_features: int = 10
) -> List[str]:
    """
    Select top features.

    Args:
        X: Features
        y: Labels
        method: Selection method
        n_features: Number of features to select

    Returns:
        List of selected feature names
    """
    from sklearn.feature_selection import mutual_info_classif

    if method == 'importance':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        importances = pd.Series(model.feature_importances_, index=X.columns)
        top_features = importances.nlargest(n_features).index.tolist()

    elif method == 'mutual_info':
        mi_scores = mutual_info_classif(X, y)
        mi_series = pd.Series(mi_scores, index=X.columns)
        top_features = mi_series.nlargest(n_features).index.tolist()

    else:
        raise ValueError(f"Unknown method: {method}")

    return top_features
