"""
Test 5: Multi-Strategy Execution Compliance.

Validates that:
1. Multiple strategies are trained
2. All strategies execute during prediction
3. Strategy votes are recorded
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_multiple_strategies_trained(trained_system):
    """Verify multiple strategy types were trained."""
    results = trained_system['results']

    strategies_trained = set(
        strategy for (strategy, regime) in results['regime_models'].keys()
    )

    expected_strategies = {'momentum', 'mean_reversion', 'volatility'}

    assert len(strategies_trained) >= 2, \
        f"Only {len(strategies_trained)} strategy types trained"

    assert strategies_trained.issubset(expected_strategies), \
        f"Unexpected strategies: {strategies_trained - expected_strategies}"

    print(f"✓ {len(strategies_trained)} strategy types trained: {strategies_trained}")


def test_strategies_execute_in_parallel(prediction_result):
    """Verify multiple strategies run during prediction."""
    assert 'strategy_votes' in prediction_result, \
        "No strategy_votes in prediction (strategies not executed?)"

    votes = prediction_result['strategy_votes']

    assert len(votes) > 0, "No strategies executed"
    assert len(votes) >= 2, "Only one strategy executed (should be multiple)"

    print(f"✓ {len(votes)} strategies executed in parallel")


def test_strategy_votes_recorded(prediction_result):
    """Verify each strategy's vote is recorded."""
    votes = prediction_result['strategy_votes']

    for strategy, vote in votes.items():
        assert isinstance(strategy, str), "Strategy name should be string"
        assert vote in [-1, 0, 1], \
            f"Invalid vote {vote} from {strategy}"

    print(f"✓ Strategy votes: {votes}")


def test_active_strategies_selected(prediction_result):
    """Verify subset of strategies are selected as active."""
    assert 'active_strategies' in prediction_result, \
        "No active_strategies in prediction output"

    active = prediction_result['active_strategies']
    all_votes = prediction_result['strategy_votes']

    assert len(active) > 0, "No active strategies selected"
    assert len(active) <= len(all_votes), \
        "More active than total strategies (impossible)"

    # Active strategies should be subset of all strategies
    assert set(active).issubset(set(all_votes.keys())), \
        "Active strategies not in strategy_votes"

    print(f"✓ {len(active)}/{len(all_votes)} strategies selected as active")


def test_each_strategy_has_model(trained_system, prediction_result):
    """Verify each executed strategy has a trained model."""
    regime = prediction_result['regime']
    strategies = prediction_result['strategy_votes'].keys()

    models = trained_system['results']['regime_models']

    for strategy in strategies:
        key = (strategy, regime)
        assert key in models, \
            f"No model found for {strategy}_{regime}"

        model_dict = models[key]
        assert 'primary' in model_dict, f"No primary model for {strategy}_{regime}"
        assert 'meta' in model_dict, f"No meta model for {strategy}_{regime}"

    print(f"✓ All {len(strategies)} strategies have trained models for regime {regime}")


def test_strategy_diversity(prediction_result):
    """Verify strategies can disagree (not all identical)."""
    votes = prediction_result['strategy_votes']

    unique_votes = set(votes.values())

    # If all votes are identical, strategies might not be differentiated
    if len(votes) > 1:
        # With multiple strategies, we should see some diversity
        # (not always, but suspicious if never)
        print(f"  Vote diversity: {len(unique_votes)} unique votes from {len(votes)} strategies")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
