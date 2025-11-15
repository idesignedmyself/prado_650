"""
Strategy ensemble coordination.
Manages all 7 strategies and combines predictions.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


@dataclass
class StrategyPrediction:
    """Prediction from a single strategy."""
    strategy_name: str
    signal: float  # -1 to 1
    confidence: float  # 0 to 1
    regime: Optional[str] = None
    metadata: Optional[Dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'strategy': self.strategy_name,
            'signal': self.signal,
            'confidence': self.confidence,
            'regime': self.regime,
            'metadata': self.metadata or {}
        }


def run_all_strategies(
    df: pd.DataFrame,
    strategy_models: Dict,
    features: pd.DataFrame,
    regime: Optional[str] = None
) -> List[StrategyPrediction]:
    """
    Run all strategies and collect predictions.

    Args:
        df: Market data DataFrame
        strategy_models: Dictionary of trained strategy models
        features: Feature matrix
        regime: Current regime (optional)

    Returns:
        List of StrategyPrediction objects
    """
    predictions = []

    # Get latest features
    latest_features = features.iloc[-1:] if len(features) > 0 else None

    if latest_features is None or latest_features.empty:
        return predictions

    # Run each strategy
    for strategy_name, model in strategy_models.items():
        try:
            # Predict
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(latest_features)[0]
                signal = proba[1] - proba[0]  # Assuming binary classification
                confidence = max(proba)
            else:
                signal = model.predict(latest_features)[0]
                confidence = abs(signal)

            prediction = StrategyPrediction(
                strategy_name=strategy_name,
                signal=signal,
                confidence=confidence,
                regime=regime
            )

            predictions.append(prediction)

        except Exception as e:
            print(f"Error running strategy {strategy_name}: {e}")

    return predictions


def aggregate_strategy_predictions(
    predictions: List[StrategyPrediction],
    method: str = 'weighted_average'
) -> StrategyPrediction:
    """
    Aggregate predictions from multiple strategies.

    Args:
        predictions: List of strategy predictions
        method: Aggregation method ('weighted_average', 'majority_vote', 'best_confidence')

    Returns:
        Aggregated StrategyPrediction
    """
    if not predictions:
        return StrategyPrediction(
            strategy_name='ensemble',
            signal=0.0,
            confidence=0.0
        )

    if method == 'weighted_average':
        # Weight by confidence
        total_weight = sum(p.confidence for p in predictions)
        if total_weight > 0:
            weighted_signal = sum(p.signal * p.confidence for p in predictions) / total_weight
            avg_confidence = total_weight / len(predictions)
        else:
            weighted_signal = 0.0
            avg_confidence = 0.0

        return StrategyPrediction(
            strategy_name='ensemble_weighted',
            signal=weighted_signal,
            confidence=avg_confidence,
            metadata={'num_strategies': len(predictions)}
        )

    elif method == 'majority_vote':
        # Vote based on signal direction
        votes = [1 if p.signal > 0 else -1 if p.signal < 0 else 0 for p in predictions]
        majority_signal = 1 if sum(votes) > 0 else -1 if sum(votes) < 0 else 0

        # Confidence is proportion of agreement
        confidence = abs(sum(votes)) / len(votes) if votes else 0

        return StrategyPrediction(
            strategy_name='ensemble_majority',
            signal=float(majority_signal),
            confidence=confidence,
            metadata={'votes': votes}
        )

    elif method == 'best_confidence':
        # Take prediction with highest confidence
        best = max(predictions, key=lambda p: p.confidence)
        return StrategyPrediction(
            strategy_name=f'ensemble_best_{best.strategy_name}',
            signal=best.signal,
            confidence=best.confidence,
            metadata={'source': best.strategy_name}
        )

    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def filter_predictions_by_regime(
    predictions: List[StrategyPrediction],
    regime: str,
    regime_strategy_map: Dict[str, List[str]]
) -> List[StrategyPrediction]:
    """
    Filter predictions to only include strategies suitable for current regime.

    Args:
        predictions: List of all predictions
        regime: Current regime
        regime_strategy_map: Mapping of regime to preferred strategies

    Returns:
        Filtered list of predictions
    """
    if regime not in regime_strategy_map:
        return predictions

    preferred_strategies = regime_strategy_map[regime]

    filtered = [
        p for p in predictions
        if p.strategy_name in preferred_strategies
    ]

    return filtered if filtered else predictions


def rank_strategies_by_performance(
    predictions: List[StrategyPrediction],
    historical_performance: Dict[str, float]
) -> List[StrategyPrediction]:
    """
    Rank strategies by historical performance.

    Args:
        predictions: List of predictions
        historical_performance: Dict of strategy_name -> performance score

    Returns:
        Sorted list of predictions (best first)
    """
    def get_score(pred):
        return historical_performance.get(pred.strategy_name, 0.0)

    return sorted(predictions, key=get_score, reverse=True)


def get_strategy_consensus(predictions: List[StrategyPrediction]) -> Dict:
    """
    Calculate consensus metrics from strategy predictions.

    Args:
        predictions: List of predictions

    Returns:
        Dictionary with consensus metrics
    """
    if not predictions:
        return {
            'consensus_signal': 0.0,
            'consensus_strength': 0.0,
            'agreement': 0.0,
            'num_strategies': 0
        }

    signals = [p.signal for p in predictions]
    confidences = [p.confidence for p in predictions]

    # Consensus signal (average)
    consensus_signal = np.mean(signals)

    # Consensus strength (how aligned are signals)
    signal_std = np.std(signals)
    consensus_strength = 1.0 - min(signal_std, 1.0)

    # Agreement (percentage with same direction)
    direction = 1 if consensus_signal > 0 else -1
    agreement = sum(1 for s in signals if np.sign(s) == direction) / len(signals)

    return {
        'consensus_signal': consensus_signal,
        'consensus_strength': consensus_strength,
        'agreement': agreement,
        'num_strategies': len(predictions),
        'avg_confidence': np.mean(confidences)
    }


def select_top_strategies(
    predictions: List[StrategyPrediction],
    n: int = 3,
    criterion: str = 'confidence'
) -> List[StrategyPrediction]:
    """
    Select top N strategies based on criterion.

    Args:
        predictions: List of predictions
        n: Number of strategies to select
        criterion: Selection criterion ('confidence', 'signal_strength')

    Returns:
        Top N predictions
    """
    if criterion == 'confidence':
        sorted_preds = sorted(predictions, key=lambda p: p.confidence, reverse=True)
    elif criterion == 'signal_strength':
        sorted_preds = sorted(predictions, key=lambda p: abs(p.signal), reverse=True)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    return sorted_preds[:n]
