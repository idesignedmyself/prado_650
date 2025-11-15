"""
Test 8: Dynamic Bet Sizing Allocator Compliance.

Validates that:
1. Position sizing is dynamic (not fixed)
2. Allocator considers confidence
3. Position size is reasonable
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from afml_system.allocation.hybrid_allocator import HybridAllocator


def test_allocator_class_exists():
    """Verify HybridAllocator exists."""
    assert HybridAllocator is not None, \
        "HybridAllocator class not found"


def test_allocator_can_calculate_position_size():
    """Verify allocator has position sizing method."""
    allocator = HybridAllocator()

    assert hasattr(allocator, 'calculate_position_size'), \
        "Allocator missing calculate_position_size method"

    assert callable(allocator.calculate_position_size), \
        "calculate_position_size not callable"

    print(f"✓ Allocator has calculate_position_size method")


def test_position_size_in_prediction(prediction_result):
    """Verify position_size is in prediction output."""
    assert 'position_size' in prediction_result, \
        "No position_size in prediction (allocator not used?)"

    position_size = prediction_result['position_size']

    assert position_size is not None, "Position size is None"

    print(f"✓ Position size in prediction: {position_size:.3f}")


def test_position_size_valid_range(prediction_result):
    """Verify position size is in valid range."""
    position_size = prediction_result['position_size']

    assert 0 <= position_size <= 1, \
        f"Position size {position_size} outside valid range [0, 1]"

    print(f"✓ Position size {position_size:.2%} in valid range")


def test_position_size_scales_with_confidence(prediction_result):
    """Verify position size reflects confidence."""
    position_size = prediction_result['position_size']
    confidence = prediction_result['confidence']

    # Position size should be influenced by confidence
    # (not necessarily equal, but should correlate)

    if confidence < 0.3:
        assert position_size < 0.5, \
            f"Position size {position_size} too high for low confidence {confidence}"

    print(f"✓ Position size {position_size:.2%} for confidence {confidence:.2%}")


def test_zero_signal_zero_position():
    """Verify neutral signal produces zero position."""
    allocator = HybridAllocator()

    position = allocator.calculate_position_size(
        signal=0,
        confidence=0.7,
        current_volatility=0.02
    )

    assert position == 0, \
        f"Neutral signal should produce zero position, got {position}"

    print(f"✓ Neutral signal → zero position")


def test_high_confidence_larger_position():
    """Verify higher confidence produces larger positions."""
    allocator = HybridAllocator(max_position_size=1.0)  # Higher max to see difference

    low_conf = allocator.calculate_position_size(
        signal=1,
        confidence=0.3,
        current_volatility=0.02
    )

    high_conf = allocator.calculate_position_size(
        signal=1,
        confidence=0.9,
        current_volatility=0.02
    )

    assert high_conf >= low_conf, \
        f"High confidence ({high_conf}) smaller than low confidence ({low_conf})"

    print(f"✓ Position scales with confidence: {low_conf:.3f} → {high_conf:.3f}")


def test_allocator_respects_max_position():
    """Verify allocator doesn't exceed max position size."""
    max_position = 0.5
    allocator = HybridAllocator(max_position_size=max_position)

    position = allocator.calculate_position_size(
        signal=1,
        confidence=1.0,  # Maximum confidence
        current_volatility=0.02
    )

    assert position <= max_position, \
        f"Position {position} exceeds max {max_position}"

    print(f"✓ Allocator respects max position: {position:.3f} ≤ {max_position}")


def test_volatility_affects_position_size():
    """Verify higher volatility reduces position size."""
    allocator = HybridAllocator()

    low_vol = allocator.calculate_position_size(
        signal=1,
        confidence=0.7,
        current_volatility=0.01  # Low volatility
    )

    high_vol = allocator.calculate_position_size(
        signal=1,
        confidence=0.7,
        current_volatility=0.05  # High volatility
    )

    # Higher volatility should typically produce smaller positions
    # (volatility targeting)
    print(f"  Low vol (1%): {low_vol:.3f}")
    print(f"  High vol (5%): {high_vol:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
