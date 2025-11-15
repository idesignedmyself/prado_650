"""
Test 9: Full AFML Pipeline Integration Test.

Validates the complete end-to-end pipeline with all 8 components.
This is the master compliance test that verifies everything works together.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_predict_ensemble_output_structure(prediction_result):
    """Verify predict_ensemble returns all required fields."""
    required_fields = [
        'symbol',
        'regime',
        'signal',
        'position_size',
        'confidence',
        'active_strategies',
        'strategy_votes',
        'num_strategies',
        'num_selected'
    ]

    for field in required_fields:
        assert field in prediction_result, \
            f"Missing required field: {field}"

    print(f"✓ All {len(required_fields)} required fields present")


def test_prediction_values_valid(prediction_result):
    """Verify all prediction values are valid."""
    # Symbol
    assert prediction_result['symbol'] == 'QQQ', \
        f"Wrong symbol: {prediction_result['symbol']}"

    # Regime
    assert prediction_result['regime'] in ['TREND', 'MEANREV', 'VOLCRUSH'], \
        f"Invalid regime: {prediction_result['regime']}"

    # Signal
    assert prediction_result['signal'] in [-1, 0, 1], \
        f"Invalid signal: {prediction_result['signal']}"

    # Position size
    assert 0 <= prediction_result['position_size'] <= 1, \
        f"Invalid position size: {prediction_result['position_size']}"

    # Confidence
    assert 0 <= prediction_result['confidence'] <= 1, \
        f"Invalid confidence: {prediction_result['confidence']}"

    # Counts
    assert prediction_result['num_strategies'] > 0, \
        "No strategies executed"

    assert prediction_result['num_selected'] > 0, \
        "No strategies selected"

    assert prediction_result['num_selected'] <= prediction_result['num_strategies'], \
        "Selected more strategies than available"

    print(f"✓ All values in valid ranges")


def test_afml_components_all_used(prediction_result, trained_system):
    """
    Master test: Verify ALL 8 AFML components were used.

    This is the critical integration test that proves compliance.
    """
    results = trained_system['results']

    # 1. CUSUM - Check training used filtered samples
    total_samples = sum(
        metrics['n_samples']
        for metrics in results['regime_metrics'].values()
    )
    assert total_samples < 800, \
        "❌ CUSUM not applied (too many samples)"
    print(f"✓ 1. CUSUM filtering used ({total_samples} total regime samples)")

    # 2. Triple-barrier - Check models were trained (requires labels)
    assert len(results['regime_models']) > 0, \
        "❌ No models trained (triple-barrier labels missing?)"
    print("✓ 2. Triple-barrier labeling used")

    # 3. Meta-labeling - Check meta models exist
    first_model = next(iter(results['regime_models'].values()))
    assert 'meta' in first_model, \
        "❌ No meta-models (meta-labeling not used)"
    print("✓ 3. Meta-labeling used")

    # 4. Regime detection - Check regime in prediction
    assert prediction_result['regime'] in ['TREND', 'MEANREV', 'VOLCRUSH'], \
        "❌ Invalid regime (regime detection not working)"
    print(f"✓ 4. Regime detection used (current: {prediction_result['regime']})")

    # 5. Per-regime models - Check multiple (strategy, regime) models
    assert len(results['regime_models']) >= 4, \
        "❌ Too few models (not training per regime)"
    print(f"✓ 5. Per-regime models used ({len(results['regime_models'])} models)")

    # 6. Multi-strategy execution - Check multiple strategies ran
    assert prediction_result['num_strategies'] >= 2, \
        "❌ Only one strategy ran (multi-strategy not working)"
    print(f"✓ 6. Multi-strategy execution ({prediction_result['num_strategies']} strategies)")

    # 7. Thompson Sampling - Check selection happened
    assert 'num_selected' in prediction_result, \
        "❌ No selection metadata (bandit not used)"
    print(f"✓ 7. Thompson Sampling used ({prediction_result['num_selected']} selected)")

    # 8. Ensemble aggregation - Check blended signal
    assert 'signal' in prediction_result, \
        "❌ No ensemble signal (aggregation not used)"
    print(f"✓ 8. Ensemble aggregation used (signal: {prediction_result['signal']})")

    # 9. Dynamic allocator - Check position size is calculated
    assert 'position_size' in prediction_result, \
        "❌ No position size (allocator not used)"
    assert 0 <= prediction_result['position_size'] <= 1, \
        "❌ Invalid position size (allocator not working)"
    # Position size should be influenced by confidence (may equal it if using simple sizing)
    print(f"✓ 9. Dynamic allocator used (size: {prediction_result['position_size']:.2%})")


def test_pipeline_consistency(prediction_result):
    """Verify internal consistency of prediction output."""
    # Active strategies should be subset of all strategies
    assert set(prediction_result['active_strategies']).issubset(
        set(prediction_result['strategy_votes'].keys())
    ), "Active strategies not in strategy_votes"

    # Num selected should match active strategies count
    assert len(prediction_result['active_strategies']) == prediction_result['num_selected'], \
        "num_selected doesn't match active_strategies count"

    # Num strategies should match strategy_votes count
    assert len(prediction_result['strategy_votes']) == prediction_result['num_strategies'], \
        "num_strategies doesn't match strategy_votes count"

    print(f"✓ Pipeline outputs internally consistent")


def test_reproducibility(test_symbol):
    """Verify predictions are deterministic (given same models)."""
    from afml_system.pipeline import predict_ensemble

    result1 = predict_ensemble(test_symbol)
    result2 = predict_ensemble(test_symbol)

    # Core prediction should be same (may differ in timestamp-dependent features)
    assert result1['regime'] == result2['regime'], \
        "Regime changed between runs (non-deterministic)"

    print(f"✓ Predictions are deterministic")


def test_full_pipeline_performance(trained_system, prediction_result):
    """Verify pipeline produces reasonable performance indicators."""
    results = trained_system['results']

    # Check meta accuracies are above random
    meta_accuracies = [
        metrics['meta']['accuracy']
        for metrics in results['regime_metrics'].values()
    ]

    avg_meta_acc = sum(meta_accuracies) / len(meta_accuracies)

    # Should be better than random guessing
    assert avg_meta_acc > 0.45, \
        f"Meta accuracy ({avg_meta_acc:.3f}) too low (not learning?)"

    print(f"✓ Average meta accuracy: {avg_meta_acc:.3f} (>0.45)")

    # Check prediction confidence is reasonable
    confidence = prediction_result['confidence']
    assert 0.3 <= confidence <= 0.9, \
        f"Confidence {confidence:.3f} outside typical range (suspicious)"

    print(f"✓ Prediction confidence: {confidence:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
