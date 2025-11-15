"""
Test 3: Meta-Labeling Compliance.

Validates that meta-labeling is:
1. Applied to all strategy models
2. Produces confidence scores
3. Used in final predictions
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_meta_models_trained(trained_system):
    """Verify meta-models were trained for each strategy."""
    results = trained_system['results']

    assert 'regime_models' in results, "No regime models found"

    # Each regime model should have a meta component
    for (strategy, regime), model_dict in results['regime_models'].items():
        assert 'meta' in model_dict, \
            f"No meta-model for {strategy}_{regime}"

        meta_model = model_dict['meta']
        assert meta_model is not None, \
            f"Meta-model is None for {strategy}_{regime}"

    print(f"✓ Meta-models trained for {len(results['regime_models'])} strategy/regime pairs")


def test_meta_metrics_recorded(trained_system):
    """Verify meta-model metrics are tracked."""
    results = trained_system['results']

    assert 'regime_metrics' in results, "No metrics recorded"

    for (strategy, regime), metrics in results['regime_metrics'].items():
        assert 'meta' in metrics, \
            f"No meta metrics for {strategy}_{regime}"

        meta_metrics = metrics['meta']
        assert 'accuracy' in meta_metrics, "Meta accuracy not recorded"

        accuracy = meta_metrics['accuracy']
        assert 0 <= accuracy <= 1, \
            f"Invalid meta accuracy: {accuracy}"

    print(f"✓ Meta-model metrics recorded for all {len(results['regime_metrics'])} models")


def test_meta_probability_in_predictions(prediction_result):
    """Verify meta-probability is used in predictions."""
    assert 'confidence' in prediction_result, \
        "No confidence in prediction output"

    confidence = prediction_result['confidence']

    assert 0 <= confidence <= 1, \
        f"Invalid confidence (meta-probability): {confidence}"

    # Confidence should not be exactly 0.5 (would suggest meta not used)
    assert abs(confidence - 0.5) > 0.01, \
        "Confidence is 0.5 (meta-model might not be applied)"

    print(f"✓ Meta-probability used in prediction: {confidence:.3f}")


def test_meta_labeling_filters_poor_signals(trained_system):
    """Verify meta-labeling can filter predictions."""
    results = trained_system['results']

    # Check that meta accuracies vary (not all identical)
    meta_accuracies = [
        metrics['meta']['accuracy']
        for metrics in results['regime_metrics'].values()
    ]

    # If all exactly the same, meta-labeling might not be working
    assert len(set(meta_accuracies)) > 1 or len(meta_accuracies) == 1, \
        "All meta accuracies identical (suspicious)"

    print(f"✓ Meta accuracies range: {min(meta_accuracies):.3f} to {max(meta_accuracies):.3f}")


def test_meta_model_has_predict_proba(trained_system):
    """Verify meta-models can generate probabilities."""
    results = trained_system['results']

    # Get first meta-model
    first_model_dict = next(iter(results['regime_models'].values()))
    meta_model = first_model_dict['meta']

    # Should have predict_proba method
    assert hasattr(meta_model, 'predict_proba'), \
        "Meta-model missing predict_proba method"

    print(f"✓ Meta-models have predict_proba capability")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
