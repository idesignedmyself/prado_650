"""
Meta-selector with 3-gate filtering.
Combines primary model, meta-model, and regime filter.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple


class MetaSelector:
    """
    Meta-selector with 3-gate filtering system.

    Gate 1: Primary model prediction
    Gate 2: Meta-model confidence filter
    Gate 3: Regime appropriateness filter
    """

    def __init__(
        self,
        primary_model: Any,
        meta_model: Any,
        regime_detector: Any,
        meta_threshold: float = 0.5,
        regime_strategy_map: Optional[Dict] = None
    ):
        """
        Initialize meta-selector.

        Args:
            primary_model: Primary prediction model
            meta_model: Meta-labeling model
            regime_detector: Regime detection model
            meta_threshold: Threshold for meta-model
            regime_strategy_map: Mapping of regimes to strategies
        """
        self.primary_model = primary_model
        self.meta_model = meta_model
        self.regime_detector = regime_detector
        self.meta_threshold = meta_threshold
        self.regime_strategy_map = regime_strategy_map or {}

    def predict(
        self,
        X: pd.DataFrame,
        strategy_name: Optional[str] = None
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Make predictions through 3-gate system.

        Args:
            X: Features
            strategy_name: Strategy name for regime filtering

        Returns:
            Tuple of (final_predictions, gate_status)
        """
        gate_status = pd.DataFrame(index=X.index)

        # Gate 1: Primary model
        primary_pred = pd.Series(
            self.primary_model.predict(X),
            index=X.index
        )
        gate_status['gate1_primary'] = primary_pred

        # Gate 2: Meta-model
        if self.meta_model is not None:
            # Get primary probabilities for meta features
            if hasattr(self.primary_model, 'predict_proba'):
                primary_proba = self.primary_model.predict_proba(X)
                X_meta = X.copy()
                X_meta['primary_pred'] = primary_pred
                X_meta['primary_proba'] = primary_proba[:, 1]

                meta_pred = pd.Series(
                    self.meta_model.predict(X_meta),
                    index=X.index
                )
            else:
                meta_pred = pd.Series(1, index=X.index)

            gate_status['gate2_meta'] = meta_pred

            # Filter predictions
            filtered_pred = primary_pred.copy()
            filtered_pred[meta_pred == 0] = 0
        else:
            filtered_pred = primary_pred
            gate_status['gate2_meta'] = 1

        # Gate 3: Regime filter
        if self.regime_detector is not None and strategy_name is not None:
            current_regime = self._get_current_regime(X)
            regime_ok = self._check_regime_compatibility(current_regime, strategy_name)

            gate_status['gate3_regime'] = regime_ok
            gate_status['regime'] = current_regime

            if not regime_ok:
                filtered_pred[:] = 0
        else:
            gate_status['gate3_regime'] = True

        gate_status['final_prediction'] = filtered_pred

        return filtered_pred, gate_status

    def _get_current_regime(self, X: pd.DataFrame) -> str:
        """Get current market regime."""
        # Simplified - would use regime detector
        return "unknown"

    def _check_regime_compatibility(
        self,
        regime: str,
        strategy_name: str
    ) -> bool:
        """Check if strategy is suitable for regime."""
        if not self.regime_strategy_map:
            return True

        if regime not in self.regime_strategy_map:
            return True

        suitable_strategies = self.regime_strategy_map[regime]
        return strategy_name in suitable_strategies

    def get_confidence(self, X: pd.DataFrame) -> pd.Series:
        """
        Get prediction confidence.

        Args:
            X: Features

        Returns:
            Confidence scores
        """
        if hasattr(self.primary_model, 'predict_proba'):
            proba = self.primary_model.predict_proba(X)
            confidence = pd.Series(proba.max(axis=1), index=X.index)
        else:
            confidence = pd.Series(0.5, index=X.index)

        return confidence

    def explain_prediction(
        self,
        X: pd.DataFrame,
        idx: int
    ) -> Dict:
        """
        Explain prediction for a sample.

        Args:
            X: Features
            idx: Sample index

        Returns:
            Explanation dictionary
        """
        X_sample = X.iloc[idx:idx+1]

        explanation = {
            'index': idx,
            'primary_prediction': self.primary_model.predict(X_sample)[0],
            'gates_passed': []
        }

        # Check each gate
        if hasattr(self.primary_model, 'predict_proba'):
            proba = self.primary_model.predict_proba(X_sample)[0]
            explanation['primary_confidence'] = proba.max()

        if self.meta_model is not None:
            meta_pred = self.meta_model.predict(X_sample)[0]
            explanation['meta_prediction'] = meta_pred
            explanation['gates_passed'].append('meta' if meta_pred == 1 else 'meta_blocked')

        return explanation
