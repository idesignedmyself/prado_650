"""
Phase 1 Demo: CUSUM Filter and Information-Driven Bars

This demo shows how to:
1. Generate CUSUM filter events from price series
2. Generate different types of information-driven bars
3. Compare bar generation methods
4. Visualize sampling results
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
    cusum_filter_symmetric,
    adaptive_cusum_filter,
    dollar_bars,
    volume_bars,
    volatility_bars,
    imbalance_bars,
    get_optimal_bar_threshold,
)


def demo_cusum_filter():
    """Demonstrate CUSUM filtering."""
    print("=" * 60)
    print("Phase 1: CUSUM Filter and Information-Driven Bars")
    print("=" * 60)

    print("\n1. Loading SPY data...")
    df = get_spy_data(start_date="2023-01-01", end_date="2023-06-30")
    print(f"   - Loaded {len(df)} days of SPY data")
    print(f"   - Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"   - Price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")

    # Standard CUSUM filter
    print("\n2. Applying Standard CUSUM Filter (threshold=0.01)...")
    threshold = 0.01
    events = cusum_filter(df['Close'], threshold=threshold)
    print(f"   - Detected {len(events)} events")
    if len(events) > 0:
        print(f"   - First event: {events[0]}")
        print(f"   - Last event: {events[-1]}")
        print(f"   - Average days between events: {len(df) / len(events):.1f}")

    # Symmetric CUSUM filter
    print("\n3. Applying Symmetric CUSUM Filter...")
    events_sym = cusum_filter_symmetric(df['Close'], threshold=threshold)
    print(f"   - Detected {len(events_sym)} events")
    if len(events_sym) > 0:
        print(f"   - Direction breakdown:")
        up_events = (events_sym['direction'] == 1).sum()
        down_events = (events_sym['direction'] == -1).sum()
        print(f"     - Up moves: {up_events}")
        print(f"     - Down moves: {down_events}")
        print(f"     - Avg magnitude: {events_sym['magnitude'].mean():.4f}")

    # Adaptive CUSUM filter
    print("\n4. Applying Adaptive CUSUM Filter...")
    adaptive_events = adaptive_cusum_filter(
        df['Close'], base_threshold=0.01, volatility_window=20
    )
    print(f"   - Detected {len(adaptive_events)} events with volatility adjustment")

    return df, events


def demo_information_driven_bars(df):
    """Demonstrate information-driven bar generation."""
    print("\n5. Generating Information-Driven Bars...")

    # Get optimal thresholds
    print("\n   a) Calculating optimal thresholds for ~100 bars...")
    dollar_threshold = get_optimal_bar_threshold(df, bar_type='dollar', target_bars=100)
    volume_threshold = get_optimal_bar_threshold(df, bar_type='volume', target_bars=100)
    volatility_threshold = get_optimal_bar_threshold(df, bar_type='volatility', target_bars=100)

    print(f"      - Dollar threshold: ${dollar_threshold:,.0f}")
    print(f"      - Volume threshold: {volume_threshold:,.0f} shares")
    print(f"      - Volatility threshold: {volatility_threshold:.6f}")

    # Generate dollar bars
    print("\n   b) Dollar Bars (price x volume)...")
    d_bars = dollar_bars(df, threshold=dollar_threshold)
    print(f"      - Generated {len(d_bars)} bars")
    if len(d_bars) > 0:
        print(f"      - First bar: {d_bars.index[0]} Close: ${d_bars['Close'].iloc[0]:.2f}")
        print(f"      - Bar OHLC range: ${d_bars['Low'].min():.2f} - ${d_bars['High'].max():.2f}")
        print(f"      - Avg bar volume: {d_bars['Volume'].mean():,.0f}")

    # Generate volume bars
    print("\n   c) Volume Bars (share count)...")
    v_bars = volume_bars(df, threshold=volume_threshold)
    print(f"      - Generated {len(v_bars)} bars")
    if len(v_bars) > 0:
        print(f"      - Avg volume per bar: {v_bars['Volume'].mean():,.0f}")
        print(f"      - Min volume per bar: {v_bars['Volume'].min():,.0f}")
        print(f"      - Max volume per bar: {v_bars['Volume'].max():,.0f}")

    # Generate volatility bars
    print("\n   d) Volatility Bars (absolute returns)...")
    vol_bars = volatility_bars(df, threshold=volatility_threshold)
    print(f"      - Generated {len(vol_bars)} bars")
    if len(vol_bars) > 0:
        print(f"      - Avg bar price movement: {((vol_bars['High'] - vol_bars['Low']) / vol_bars['Close']).mean():.4f}")

    # Generate imbalance bars
    print("\n   e) Imbalance Bars (tick direction)...")
    # Imbalance bars work better with higher frequency data
    imb_threshold = len(df) / 100  # Roughly 100 bars
    imb_bars = imbalance_bars(df, threshold=imb_threshold)
    print(f"      - Generated {len(imb_bars)} bars")

    return d_bars, v_bars, vol_bars, imb_bars


def demo_bar_comparison(df, d_bars, v_bars, vol_bars, imb_bars):
    """Compare bar generation methods."""
    print("\n6. Comparing Bar Generation Methods:")

    # Statistics comparison
    methods = [
        ("Original Time Series", df, f"{len(df)} bars"),
        ("Dollar Bars", d_bars, f"{len(d_bars)} bars"),
        ("Volume Bars", v_bars, f"{len(v_bars)} bars"),
        ("Volatility Bars", vol_bars, f"{len(vol_bars)} bars"),
        ("Imbalance Bars", imb_bars, f"{len(imb_bars)} bars"),
    ]

    print("\n   Bar count comparison:")
    for name, data, count in methods:
        print(f"   - {name:25s}: {count}")

    print("\n   Price movement (High-Low as % of Close):")
    if len(d_bars) > 0:
        d_bars_range = ((d_bars['High'] - d_bars['Low']) / d_bars['Close']).mean()
        print(f"   - Dollar Bars:       {d_bars_range:.4f}")
    if len(v_bars) > 0:
        v_bars_range = ((v_bars['High'] - v_bars['Low']) / v_bars['Close']).mean()
        print(f"   - Volume Bars:       {v_bars_range:.4f}")
    if len(vol_bars) > 0:
        vol_bars_range = ((vol_bars['High'] - vol_bars['Low']) / vol_bars['Close']).mean()
        print(f"   - Volatility Bars:   {vol_bars_range:.4f}")
    if len(imb_bars) > 0:
        imb_bars_range = ((imb_bars['High'] - imb_bars['Low']) / imb_bars['Close']).mean()
        print(f"   - Imbalance Bars:    {imb_bars_range:.4f}")

    # Volume consistency
    print("\n   Volume per bar (consistency):")
    if len(d_bars) > 0:
        d_vol_std = d_bars['Volume'].std() / d_bars['Volume'].mean()
        print(f"   - Dollar Bars:       {d_vol_std:.4f} (CV)")
    if len(v_bars) > 0:
        v_vol_std = v_bars['Volume'].std() / v_bars['Volume'].mean()
        print(f"   - Volume Bars:       {v_vol_std:.4f} (CV)")
    if len(vol_bars) > 0:
        vol_vol_std = vol_bars['Volume'].std() / vol_bars['Volume'].mean()
        print(f"   - Volatility Bars:   {vol_vol_std:.4f} (CV)")


if __name__ == "__main__":
    # Run CUSUM demo
    df, events = demo_cusum_filter()

    # Run information-driven bars demo
    d_bars, v_bars, vol_bars, imb_bars = demo_information_driven_bars(df)

    # Run comparison
    demo_bar_comparison(df, d_bars, v_bars, vol_bars, imb_bars)

    print("\n" + "=" * 60)
    print("Phase 1 Demo Complete!")
    print("=" * 60)
    print("\nKey Insights:")
    print("- CUSUM filter detects significant price movements")
    print("- Information-driven bars sample events instead of time")
    print("- Dollar bars balance price and volume")
    print("- Volume bars normalize by share count")
    print("- Volatility bars normalize by price movement")
    print("- Imbalance bars consider order direction")
    print("\nNext: Run phase2_demo.py for triple barrier labeling")
