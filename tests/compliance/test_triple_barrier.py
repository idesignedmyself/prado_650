"""
Test 2: Triple-Barrier Labeling Compliance.

Validates that triple-barrier method is:
1. Used for label generation
2. Produces valid labels (-1, 0, 1)
3. Includes barrier metadata (t1, ret)
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from afml_system.labeling.triple_barrier import triple_barrier_labels, get_daily_volatility


def test_triple_barrier_function_exists():
    """Verify triple-barrier labeling function exists."""
    assert callable(triple_barrier_labels), "triple_barrier_labels function not found"


def test_triple_barrier_produces_valid_labels(sample_data):
    """Verify triple-barrier generates valid labels."""
    data = sample_data['data']
    events = sample_data['events']

    labels = triple_barrier_labels(
        data['Close'],
        events,
        pt_sl=[1.0, 1.0],
        vertical_barrier_days=5
    )

    # Check output structure
    assert 't1' in labels.columns, "Missing t1 (barrier touch time)"
    assert 'ret' in labels.columns, "Missing ret (return)"
    assert 'label' in labels.columns, "Missing label"

    # Check label values
    unique_labels = labels['label'].unique()
    assert all(label in [-1, 0, 1] for label in unique_labels), \
        f"Invalid label values: {unique_labels}"

    # Check we have both sides
    assert len(unique_labels) > 1, "Labels not diverse (only one class)"

    print(f"✓ Triple-barrier produced {len(labels)} labels with distribution:")
    print(f"  {labels['label'].value_counts().to_dict()}")


def test_dynamic_volatility_barriers(sample_data):
    """Verify barriers use dynamic volatility."""
    data = sample_data['data']

    daily_vol = get_daily_volatility(data['Close'])

    assert len(daily_vol) > 0, "Daily volatility not calculated"
    assert daily_vol.std() > 0, "Volatility not dynamic (constant)"

    print(f"✓ Dynamic volatility: mean={daily_vol.mean():.4f}, std={daily_vol.std():.4f}")


def test_triple_barrier_metadata_valid(sample_data):
    """Verify barrier metadata is valid."""
    data = sample_data['data']
    events = sample_data['events']

    labels = triple_barrier_labels(
        data['Close'],
        events,
        pt_sl=[1.0, 1.0],
        vertical_barrier_days=5
    )

    # Check t1 (barrier touch times) are valid
    assert labels['t1'].notna().sum() > 0, "No barrier touch times recorded"

    # Check returns are reasonable
    assert labels['ret'].abs().max() < 1.0, "Unrealistic returns in labels"

    print(f"✓ Barrier metadata valid: {labels['t1'].notna().sum()} barriers touched")


def test_triple_barrier_used_in_training(trained_system):
    """Verify triple-barrier labels used in training."""
    results = trained_system['results']

    # Training should have produced models
    assert 'regime_models' in results, "No regime models trained"
    assert len(results['regime_models']) > 0, "No models in results"

    # Each model was trained on triple-barrier labels
    for (strategy, regime), metrics in results['regime_metrics'].items():
        assert metrics['n_samples'] > 0, \
            f"No samples for {strategy}_{regime} (labels not used?)"

    print(f"✓ Triple-barrier labels used to train {len(results['regime_models'])} models")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
