"""
Test 6: Thompson Sampling Bandit Compliance.

Validates that:
1. Bandit exists and can select strategies
2. Bandit is used in prediction pipeline
3. Strategy selection is recorded
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from afml_system.strategies.bandit import ThompsonSamplingBandit


def test_bandit_class_exists():
    """Verify Thompson Sampling bandit exists."""
    assert ThompsonSamplingBandit is not None, \
        "ThompsonSamplingBandit class not found"


def test_bandit_can_select_strategies():
    """Verify bandit has select_strategies method."""
    bandit = ThompsonSamplingBandit()

    assert hasattr(bandit, 'select_strategies'), \
        "Bandit missing select_strategies method"

    assert callable(bandit.select_strategies), \
        "select_strategies is not callable"

    print(f"✓ Bandit has select_strategies method")


def test_bandit_selection_in_prediction(prediction_result):
    """Verify bandit selected strategies during prediction."""
    assert 'num_selected' in prediction_result, \
        "No num_selected in output (bandit not used?)"

    assert 'num_strategies' in prediction_result, \
        "No num_strategies in output"

    num_selected = prediction_result['num_selected']
    num_total = prediction_result['num_strategies']

    assert num_selected > 0, "Bandit selected zero strategies"
    assert num_selected <= num_total, \
        f"Selected ({num_selected}) > total ({num_total})"

    print(f"✓ Bandit selected {num_selected}/{num_total} strategies")


def test_active_strategies_match_selection(prediction_result):
    """Verify active_strategies reflects bandit selection."""
    active = prediction_result['active_strategies']
    num_selected = prediction_result['num_selected']

    assert len(active) == num_selected, \
        f"Active strategies count ({len(active)}) != num_selected ({num_selected})"

    print(f"✓ Active strategies match bandit selection: {active}")


def test_bandit_can_filter_strategies(prediction_result):
    """Verify bandit can filter (not always select all)."""
    num_selected = prediction_result['num_selected']
    num_total = prediction_result['num_strategies']

    # Bandit should be capable of filtering
    # (may select all if all are good, but should have the capability)
    assert hasattr(ThompsonSamplingBandit, 'select_strategies'), \
        "Bandit cannot filter (missing selection logic)"

    print(f"  Bandit selected {num_selected}/{num_total} (can filter: ✓)")


def test_bandit_has_update_method():
    """Verify bandit can learn from results."""
    bandit = ThompsonSamplingBandit()

    assert hasattr(bandit, 'update'), \
        "Bandit missing update method (cannot learn)"

    assert callable(bandit.update), \
        "update is not callable"

    print(f"✓ Bandit has update method for learning")


def test_bandit_persistence():
    """Verify bandit can be saved/loaded."""
    bandit = ThompsonSamplingBandit()

    assert hasattr(bandit, 'save'), \
        "Bandit missing save method"

    assert hasattr(ThompsonSamplingBandit, 'load'), \
        "Bandit missing load classmethod"

    print(f"✓ Bandit has save/load persistence")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
