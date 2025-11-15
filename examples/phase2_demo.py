"""
Phase 2 Demo: Triple Barrier Labeling

This demo shows how to:
1. Generate labels using the triple barrier method
2. Understand horizontal, vertical, and exponential barriers
3. Calculate sample weights for class imbalance
4. Prepare labeled data for model training
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from afml_system.data import (
    get_spy_data,
    cusum_filter,
)
from afml_system.labeling import (
    triple_barrier_labels,
    get_daily_volatility,
    get_bins,
    drop_labels,
    get_sample_weights,
)


def demo_triple_barrier_labeling():
    """Demonstrate triple barrier labeling."""
    print("=" * 60)
    print("Phase 2: Triple Barrier Labeling")
    print("=" * 60)

    print("\n1. Loading SPY data...")
    df = get_spy_data(start_date="2023-01-01", end_date="2023-06-30")
    close_prices = df['Close']
    print(f"   - Loaded {len(df)} days")
    print(f"   - Date range: {df.index[0].date()} to {df.index[-1].date()}")

    # Generate events using CUSUM
    print("\n2. Detecting events with CUSUM filter...")
    events = cusum_filter(close_prices, threshold=0.01)
    print(f"   - Detected {len(events)} events")

    # Limit events for demo
    if len(events) > 50:
        events = events[:50]
        print(f"   - Using first 50 events for demo")

    # Get daily volatility
    print("\n3. Calculating daily volatility...")
    daily_vol = get_daily_volatility(close_prices, span=100)
    print(f"   - Min volatility: {daily_vol.min():.4f}")
    print(f"   - Mean volatility: {daily_vol.mean():.4f}")
    print(f"   - Max volatility: {daily_vol.max():.4f}")

    # Triple barrier parameters
    print("\n4. Triple Barrier Configuration:")
    pt_sl = [1.0, 1.0]  # profit taking and stop loss multipliers
    min_ret = 0.01  # minimum return
    vertical_barrier_days = 5

    print(f"   - Profit taking multiplier: {pt_sl[0]}")
    print(f"   - Stop loss multiplier: {pt_sl[1]}")
    print(f"   - Minimum return: {min_ret:.2%}")
    print(f"   - Vertical barrier (days): {vertical_barrier_days}")

    # Apply triple barrier
    print("\n5. Applying triple barrier labels...")
    labels_df = triple_barrier_labels(
        close=close_prices,
        events=events,
        pt_sl=pt_sl,
        min_ret=min_ret,
        num_threads=1,
        vertical_barrier_days=vertical_barrier_days,
    )

    print(f"   - Generated labels for {len(labels_df)} events")
    print(f"   - Columns: {list(labels_df.columns)}")

    # Analyze labels
    if len(labels_df) > 0:
        print("\n6. Label Distribution:")
        if 'label' in labels_df.columns:
            label_counts = labels_df['label'].value_counts().sort_index()
            for label, count in label_counts.items():
                pct = 100 * count / len(labels_df)
                label_name = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}.get(label, 'Unknown')
                print(f"   - {label_name:10s} (label={label:2d}): {count:3d} ({pct:5.1f}%)")

        # Calculate returns
        print("\n7. Return Statistics:")
        if 'ret' in labels_df.columns:
            returns = labels_df['ret']
            print(f"   - Mean return: {returns.mean():.4f}")
            print(f"   - Std return: {returns.std():.4f}")
            print(f"   - Min return: {returns.min():.4f}")
            print(f"   - Max return: {returns.max():.4f}")
            print(f"   - Positive returns: {(returns > 0).sum()} ({100 * (returns > 0).sum() / len(returns):.1f}%)")
            print(f"   - Negative returns: {(returns < 0).sum()} ({100 * (returns < 0).sum() / len(returns):.1f}%)")

    return close_prices, labels_df, daily_vol


def demo_bins_conversion(labels_df):
    """Convert labels to bins."""
    print("\n8. Converting to Bin Labels...")
    if len(labels_df) > 0:
        bins_df = get_bins(labels_df, close_prices)
        print(f"   - Generated {len(bins_df)} binned labels")
        if 'bin' in bins_df.columns:
            bin_counts = bins_df['bin'].value_counts().sort_index()
            for bin_val, count in bin_counts.items():
                pct = 100 * count / len(bins_df)
                print(f"   - Bin {bin_val}: {count} ({pct:.1f}%)")

        return bins_df

    return None


def demo_class_weighting(labels_df):
    """Demonstrate sample weighting for class imbalance."""
    print("\n9. Sample Weighting for Class Imbalance...")

    if len(labels_df) > 0:
        # Calculate sample weights
        weights = get_sample_weights(labels_df)
        print(f"   - Calculated weights for {len(weights)} samples")
        print(f"   - Weight statistics:")
        print(f"     - Min weight: {weights.min():.4f}")
        print(f"     - Mean weight: {weights.mean():.4f}")
        print(f"     - Max weight: {weights.max():.4f}")
        print(f"     - Std weight: {weights.std():.4f}")

        # Show weight by label
        if 'label' in labels_df.columns:
            print(f"\n   - Weights by label:")
            for label in sorted(labels_df['label'].unique()):
                label_weights = weights[labels_df['label'] == label]
                label_name = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}.get(label, 'Unknown')
                print(f"     - {label_name:10s}: mean={label_weights.mean():.4f}, std={label_weights.std():.4f}")

        return weights

    return None


def demo_data_filtering(labels_df):
    """Demonstrate dropping rare labels."""
    print("\n10. Filtering Rare Labels...")

    if len(labels_df) > 0:
        print(f"    - Original label distribution:")
        label_counts = labels_df['label'].value_counts()
        for label, count in label_counts.items():
            pct = 100 * count / len(labels_df)
            print(f"      - Label {label}: {count} ({pct:.1f}%)")

        # Drop rare labels
        filtered_df = drop_labels(labels_df, min_pct=0.05)
        print(f"\n    - After filtering (min 5%):")
        print(f"      - Rows: {len(labels_df)} -> {len(filtered_df)}")

        if len(filtered_df) > 0:
            filtered_counts = filtered_df['label'].value_counts()
            for label, count in filtered_counts.items():
                pct = 100 * count / len(filtered_df)
                print(f"      - Label {label}: {count} ({pct:.1f}%)")

        return filtered_df

    return None


if __name__ == "__main__":
    # Run triple barrier demo
    close_prices, labels_df, daily_vol = demo_triple_barrier_labeling()

    # Run bins conversion
    bins_df = demo_bins_conversion(labels_df)

    # Run weighting demo
    weights = demo_class_weighting(labels_df)

    # Run filtering demo
    filtered_df = demo_data_filtering(labels_df)

    print("\n" + "=" * 60)
    print("Phase 2 Demo Complete!")
    print("=" * 60)
    print("\nKey Insights:")
    print("- Triple barrier creates labels from events:")
    print("  * Horizontal barriers: profit target and stop loss")
    print("  * Vertical barrier: time limit")
    print("  * Whichever is hit first determines the label")
    print("- Labels are: 1 (profit), 0 (neutral), -1 (loss)")
    print("- Sample weights handle class imbalance")
    print("- Rare labels can be dropped to focus on meaningful classes")
    print("\nNext: Run phase3_demo.py for feature building")
