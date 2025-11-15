"""
Purged K-Fold cross-validation.
Implements purging and embargo to prevent leakage in time series.
"""
import numpy as np
import pandas as pd
from typing import Generator, Tuple
from sklearn.model_selection import KFold


class PurgedKFold(KFold):
    """
    K-Fold cross-validation with purging and embargo.

    Extends sklearn's KFold to handle overlapping samples in financial data.
    Purges training samples that overlap with test samples.
    """

    def __init__(
        self,
        n_splits: int = 5,
        samples_info_sets: pd.Series = None,
        pct_embargo: float = 0.01
    ):
        """
        Initialize Purged K-Fold.

        Args:
            n_splits: Number of folds
            samples_info_sets: Series mapping sample index to end time (t1)
            pct_embargo: Percentage of samples to embargo after test set
        """
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.samples_info_sets = samples_info_sets
        self.pct_embargo = pct_embargo

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        groups=None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate purged train/test splits.

        Args:
            X: Feature matrix
            y: Labels (not used, for compatibility)
            groups: Groups (not used, for compatibility)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        if self.samples_info_sets is None:
            # Fall back to standard KFold
            yield from super().split(X)
            return

        indices = np.arange(len(X))

        # Calculate embargo size
        embargo_size = int(len(X) * self.pct_embargo)

        # Get fold splits
        for train_idx, test_idx in super().split(X):
            # Purge training set
            train_idx = self._purge_train_set(
                train_idx,
                test_idx,
                X.index,
                embargo_size
            )

            yield train_idx, test_idx

    def _purge_train_set(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        index: pd.Index,
        embargo_size: int
    ) -> np.ndarray:
        """
        Purge training set to remove overlapping samples.

        Args:
            train_idx: Training indices
            test_idx: Test indices
            index: DataFrame index
            embargo_size: Number of samples to embargo

        Returns:
            Purged training indices
        """
        # Get test set time range
        test_times = index[test_idx]
        test_start = test_times[0]
        test_end = test_times[-1]

        # Purge training samples that overlap with test set
        purged_train = []

        for idx in train_idx:
            sample_time = index[idx]

            # Get sample end time if available
            if sample_time in self.samples_info_sets.index:
                sample_end = self.samples_info_sets.loc[sample_time]

                # Check for overlap
                if sample_end < test_start:
                    # Sample ends before test starts - OK
                    purged_train.append(idx)
                elif sample_time > test_end:
                    # Sample starts after test ends - check embargo
                    if idx > test_idx[-1] + embargo_size:
                        purged_train.append(idx)
                # else: overlaps with test set - purge
            else:
                # No end time info - use simple time check
                if sample_time < test_start or sample_time > test_end + pd.Timedelta(days=embargo_size):
                    purged_train.append(idx)

        return np.array(purged_train)


def get_train_times(
    samples_info_sets: pd.Series,
    test_times: pd.Index
) -> pd.Index:
    """
    Get training times that don't overlap with test times.

    Args:
        samples_info_sets: Series with sample end times
        test_times: Test set times

    Returns:
        Training times
    """
    test_start = test_times[0]
    test_end = test_times[-1]

    train_times = []

    for sample_time, sample_end in samples_info_sets.items():
        # Include if sample doesn't overlap with test
        if sample_end < test_start or sample_time > test_end:
            train_times.append(sample_time)

    return pd.Index(train_times)


def cv_score(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series = None,
    cv_gen: PurgedKFold = None,
    scoring: str = 'accuracy'
) -> dict:
    """
    Cross-validation with purged K-Fold.

    Args:
        model: Model to evaluate
        X: Features
        y: Labels
        sample_weight: Sample weights
        cv_gen: PurgedKFold generator
        scoring: Scoring metric

    Returns:
        Dictionary with CV results
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    if cv_gen is None:
        cv_gen = PurgedKFold(n_splits=5)

    scores = []
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(cv_gen.split(X)):
        # Get train/test data
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Get sample weights if provided
        if sample_weight is not None:
            sw_train = sample_weight.iloc[train_idx]
        else:
            sw_train = None

        # Train model
        if sw_train is not None:
            model.fit(X_train, y_train, sample_weight=sw_train)
        else:
            model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Calculate metrics
        if scoring == 'accuracy':
            score = accuracy_score(y_test, y_pred)
        elif scoring == 'f1':
            score = f1_score(y_test, y_pred, average='weighted')
        elif scoring == 'precision':
            score = precision_score(y_test, y_pred, average='weighted')
        elif scoring == 'recall':
            score = recall_score(y_test, y_pred, average='weighted')
        else:
            score = accuracy_score(y_test, y_pred)

        scores.append(score)

        fold_metrics.append({
            'fold': fold,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'score': score
        })

    return {
        'scores': scores,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'fold_metrics': fold_metrics
    }
