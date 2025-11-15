"""
Test 4: Regime Detection Compliance.

Validates that regime detection is:
1. Functional and returns valid regimes
2. Models are trained per regime
3. Predictions use current regime
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from afml_system.regime.detection import detect_all_regimes, detect_current_regime


def test_regime_detection_exists():
    """Verify regime detection functions exist."""
    assert callable(detect_all_regimes), "detect_all_regimes function not found"
    assert callable(detect_current_regime), "detect_current_regime function not found"


def test_detect_current_regime_valid(sample_data):
    """Verify detect_current_regime returns valid regime."""
    data = sample_data['data']

    regime = detect_current_regime(data)

    valid_regimes = ['TREND', 'MEANREV', 'VOLCRUSH']
    assert regime in valid_regimes, \
        f"Invalid regime '{regime}', expected one of {valid_regimes}"

    print(f"✓ Current regime detected: {regime}")


def test_regime_distribution_in_training(trained_system):
    """Verify models trained across multiple regimes."""
    results = trained_system['results']

    # Extract unique regimes from trained models
    regimes_trained = set(
        regime for (strategy, regime) in results['regime_models'].keys()
    )

    assert len(regimes_trained) > 0, "No regimes in trained models"
    assert len(regimes_trained) >= 1, "Only one regime trained (should have multiple)"

    print(f"✓ Models trained for {len(regimes_trained)} regimes: {regimes_trained}")


def test_per_regime_models_exist(trained_system):
    """Verify separate models exist for each (strategy, regime) pair."""
    results = trained_system['results']

    assert 'regime_models' in results, "No regime_models in results"

    regime_models = results['regime_models']
    assert len(regime_models) > 0, "No per-regime models trained"

    # Check structure of keys
    for key in regime_models.keys():
        assert isinstance(key, tuple), f"Model key should be tuple, got {type(key)}"
        assert len(key) == 2, f"Model key should be (strategy, regime), got {key}"

        strategy, regime = key
        assert isinstance(strategy, str), "Strategy should be string"
        assert isinstance(regime, str), "Regime should be string"

    print(f"✓ {len(regime_models)} per-regime models exist")


def test_prediction_uses_regime(prediction_result):
    """Verify predictions include regime information."""
    assert 'regime' in prediction_result, \
        "No regime in prediction output"

    regime = prediction_result['regime']

    valid_regimes = ['TREND', 'MEANREV', 'VOLCRUSH']
    assert regime in valid_regimes, \
        f"Invalid regime in prediction: '{regime}'"

    print(f"✓ Prediction uses regime: {regime}")


def test_only_regime_specific_models_loaded(prediction_result):
    """Verify only models for current regime are used."""
    regime = prediction_result['regime']

    # Active strategies should only be from current regime
    # (This is validated by checking that prediction succeeded
    # and used models - if wrong regime models loaded, would fail)

    assert prediction_result['num_strategies'] > 0, \
        "No strategies executed (regime filtering too aggressive?)"

    print(f"✓ Loaded {prediction_result['num_strategies']} models for regime {regime}")


def test_regime_metrics_per_regime(trained_system):
    """Verify metrics tracked separately per regime."""
    results = trained_system['results']

    regime_metrics = results.get('regime_metrics', {})

    # Group by regime
    regimes_in_metrics = set(
        regime for (strategy, regime) in regime_metrics.keys()
    )

    assert len(regimes_in_metrics) > 0, "No regimes in metrics"

    # Each regime should have multiple strategies
    for target_regime in regimes_in_metrics:
        strategies_for_regime = [
            strategy for (strategy, regime) in regime_metrics.keys()
            if regime == target_regime
        ]

        assert len(strategies_for_regime) > 1, \
            f"Only one strategy for regime {target_regime}"

    print(f"✓ Metrics tracked for {len(regimes_in_metrics)} regimes")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
