"""
Test 1: CUSUM Filtering Usage Compliance.

Validates that CUSUM filtering is:
1. Actually applied during data preparation
2. Reduces sample count (noise reduction)
3. Events are used in feature building
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from afml_system.data.fetch import prepare_training_data
from afml_system.data.cusum import cusum_filter
from afml_system.features.feature_union import build_feature_matrix


def test_cusum_filter_exists():
    """Verify CUSUM filter function exists."""
    assert callable(cusum_filter), "cusum_filter function not found"


def test_cusum_reduces_samples(sample_data):
    """Verify CUSUM actually reduces sample count."""
    data = sample_data['data']
    events = sample_data['events']

    assert len(events) < len(data), \
        f"CUSUM did not reduce samples: {len(events)} events vs {len(data)} bars"

    reduction_pct = (1 - len(events) / len(data)) * 100
    assert reduction_pct > 10, \
        f"CUSUM reduction too small: {reduction_pct:.1f}% (expected >10%)"

    print(f"✓ CUSUM reduced samples by {reduction_pct:.1f}%")


def test_cusum_events_used_in_features(sample_data):
    """Verify features are built at CUSUM event timestamps."""
    data = sample_data['data']
    events = sample_data['events']

    # Build features with events
    features = build_feature_matrix(data, events=events)

    # Features should have same length as events
    assert len(features) == len(events), \
        f"Features ({len(features)}) not aligned with events ({len(events)})"

    # Features index should match events
    assert features.index.equals(events), \
        "Features index does not match CUSUM events"

    print(f"✓ Features built at {len(features)} CUSUM event timestamps")


def test_prepare_training_data_returns_events():
    """Verify prepare_training_data returns events tuple."""
    result = prepare_training_data(
        "QQQ",
        start_date='2023-01-01',
        end_date='2023-03-31',
        use_cusum=True
    )

    assert isinstance(result, tuple), \
        "prepare_training_data should return (data, events) tuple"

    data, events = result

    assert len(events) > 0, "No events returned"
    assert len(events) < len(data), "Events not filtered"

    print(f"✓ prepare_training_data returns ({len(data)} bars, {len(events)} events)")


def test_cusum_threshold_dynamic():
    """Verify CUSUM uses dynamic volatility-based threshold."""
    data, events = prepare_training_data(
        "QQQ",
        start_date='2023-01-01',
        end_date='2023-03-31',
        use_cusum=True,
        cusum_threshold=None  # Should use dynamic threshold
    )

    assert len(events) > 0, "Dynamic threshold produced no events"

    # With dynamic threshold, should get reasonable reduction
    reduction = (1 - len(events) / len(data)) * 100
    assert 20 < reduction < 80, \
        f"Dynamic threshold reduction ({reduction:.1f}%) outside expected range"

    print(f"✓ Dynamic CUSUM threshold produced {reduction:.1f}% reduction")


def test_cusum_used_in_training(trained_system):
    """Verify CUSUM is used in the training pipeline."""
    results = trained_system['results']

    # Check that models were trained on reduced sample set
    if 'regime_metrics' in results:
        total_samples = sum(
            metrics['n_samples']
            for metrics in results['regime_metrics'].values()
        )

        # Total samples should be less than full data range
        # (2 years of daily data = ~500 bars, with CUSUM ~200 events)
        # Multiple regimes means samples counted multiple times
        assert total_samples < 800, \
            f"Training samples ({total_samples}) suggest CUSUM not applied"

        print(f"✓ Training used {total_samples} CUSUM-filtered samples")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
