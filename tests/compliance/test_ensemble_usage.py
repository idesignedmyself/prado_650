"""
Test 7: Ensemble Aggregation Compliance.

Validates that:
1. Predictions are aggregated across strategies
2. Conflict-aware blending is used
3. Final signal is produced
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from afml_system.strategies.ensemble import aggregate_strategy_predictions, Prediction


def test_aggregate_function_exists():
    """Verify aggregation function exists."""
    assert callable(aggregate_strategy_predictions), \
        "aggregate_strategy_predictions function not found"


def test_prediction_class_exists():
    """Verify Prediction dataclass exists."""
    assert Prediction is not None, "Prediction class not found"

    # Check required fields
    required_fields = ['strategy_name', 'side', 'probability', 'meta_probability', 'regime']

    pred = Prediction(
        strategy_name='test',
        side=1.0,
        probability=0.6,
        meta_probability=0.7,
        regime='TREND'
    )

    for field in required_fields:
        assert hasattr(pred, field), f"Prediction missing field: {field}"

    print(f"✓ Prediction class has all required fields")


def test_ensemble_signal_in_prediction(prediction_result):
    """Verify ensemble produces final signal."""
    assert 'signal' in prediction_result, \
        "No ensemble signal in prediction output"

    signal = prediction_result['signal']

    assert signal in [-1, 0, 1], \
        f"Invalid ensemble signal: {signal}"

    print(f"✓ Ensemble signal: {signal}")


def test_conflict_aware_aggregation():
    """Verify conflict-aware aggregation reduces confidence on disagreement."""
    # Create conflicting predictions
    predictions = [
        Prediction('momentum', side=1, probability=0.6, meta_probability=0.7, regime='TREND'),
        Prediction('mean_rev', side=-1, probability=0.6, meta_probability=0.7, regime='TREND'),
        Prediction('volatility', side=1, probability=0.6, meta_probability=0.7, regime='TREND')
    ]

    result = aggregate_strategy_predictions(predictions, method='conflict_aware')

    assert result is not None, "Aggregation returned None"
    assert hasattr(result, 'meta_probability'), "Result missing meta_probability"

    # With conflict (2 long, 1 short), confidence should be reduced
    # (conflict penalty of 0.5 applied)
    assert result.meta_probability < 0.7, \
        "Confidence not reduced despite conflict"

    print(f"✓ Conflict-aware reduced confidence to {result.meta_probability:.3f}")


def test_unanimous_agreement_boosts_confidence():
    """Verify unanimous agreement doesn't penalize confidence."""
    # Create agreeing predictions
    predictions = [
        Prediction('momentum', side=1, probability=0.6, meta_probability=0.6, regime='TREND'),
        Prediction('mean_rev', side=1, probability=0.6, meta_probability=0.6, regime='TREND'),
        Prediction('volatility', side=1, probability=0.6, meta_probability=0.6, regime='TREND')
    ]

    result = aggregate_strategy_predictions(predictions, method='conflict_aware')

    # No conflict, so no penalty (or boost)
    # Confidence should be close to average input
    assert 0.5 <= result.meta_probability <= 0.7, \
        f"Unexpected confidence with agreement: {result.meta_probability}"

    print(f"✓ Unanimous agreement confidence: {result.meta_probability:.3f}")


def test_ensemble_uses_strategy_votes(prediction_result):
    """Verify ensemble considers all strategy votes."""
    votes = prediction_result['strategy_votes']
    signal = prediction_result['signal']

    # Signal should be influenced by votes
    # (exact relationship depends on aggregation method)

    vote_values = list(votes.values())
    assert len(vote_values) > 0, "No votes to aggregate"

    # If all votes agree, signal should match
    if len(set(vote_values)) == 1:
        expected_signal = vote_values[0]
        assert signal == expected_signal, \
            f"Signal {signal} doesn't match unanimous vote {expected_signal}"

    print(f"✓ Ensemble signal {signal} derived from votes {votes}")


def test_confidence_in_prediction(prediction_result):
    """Verify ensemble confidence is present."""
    assert 'confidence' in prediction_result, \
        "No confidence in prediction"

    confidence = prediction_result['confidence']

    assert 0 <= confidence <= 1, \
        f"Invalid confidence: {confidence}"

    print(f"✓ Ensemble confidence: {confidence:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
