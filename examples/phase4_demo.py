"""
Phase 4 Demo: Regime Detection

This demo shows how to:
1. Detect trending vs ranging regimes using ADX
2. Detect volatility regimes
3. Detect volume regimes
4. Use regime-aware strategy selection
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from afml_system.data import get_spy_data
from afml_system.regime import (
    TrendRegimeDetector,
    VolatilityRegimeDetector,
    VolumeRegimeDetector,
    CompositeRegimeDetector,
    detect_all_regimes,
)


def demo_trend_regime():
    """Demonstrate trend regime detection."""
    print("=" * 60)
    print("Phase 4: Regime Detection")
    print("=" * 60)

    print("\n1. Loading SPY data...")
    df = get_spy_data(start_date="2023-01-01", end_date="2023-06-30")
    print(f"   - Loaded {len(df)} days")

    print("\n2. Trend Regime Detection (ADX-based)...")
    detector = TrendRegimeDetector(adx_period=14, adx_threshold=25.0)

    # Detect regimes
    regimes = detector.detect(df)
    regimes_numeric = detector.detect_numeric(df)

    print(f"   - Detected {(regimes == 'trending').sum()} trending periods")
    print(f"   - Detected {(regimes == 'ranging').sum()} ranging periods")

    # Show regime changes
    regime_changes = (regimes != regimes.shift()).sum()
    print(f"   - Number of regime changes: {regime_changes - 1}")

    # Show recent regimes
    print("\n   - Recent regimes (last 20 days):")
    for i in range(min(20, len(regimes))):
        idx = len(regimes) - 20 + i
        date = df.index[idx]
        regime = regimes.iloc[idx]
        close = df['Close'].iloc[idx]
        print(f"     {date.date()}: {regime:10s} - ${close:.2f}")

    return df, regimes, regimes_numeric


def demo_volatility_regime():
    """Demonstrate volatility regime detection."""
    print("\n3. Volatility Regime Detection...")

    df = get_spy_data(start_date="2023-01-01", end_date="2023-06-30")
    detector = VolatilityRegimeDetector(vol_period=20, vol_threshold=0.02)

    # Detect volatility regimes
    vol_regimes = detector.detect(df)

    print(f"   - Detected {(vol_regimes == 'high').sum()} high volatility periods")
    print(f"   - Detected {(vol_regimes == 'low').sum()} low volatility periods")

    # Calculate realized volatility
    returns = df['Close'].pct_change()
    realized_vol = returns.rolling(20).std()

    print(f"   - Realized volatility statistics:")
    print(f"     - Min: {realized_vol.min():.4f}")
    print(f"     - Mean: {realized_vol.mean():.4f}")
    print(f"     - Max: {realized_vol.max():.4f}")

    # Show regime distribution
    print(f"\n   - Volatility regime distribution:")
    print(f"     - High vol periods: {(vol_regimes == 'high').sum()} ({100 * (vol_regimes == 'high').sum() / len(vol_regimes):.1f}%)")
    print(f"     - Low vol periods: {(vol_regimes == 'low').sum()} ({100 * (vol_regimes == 'low').sum() / len(vol_regimes):.1f}%)")

    return vol_regimes


def demo_volume_regime():
    """Demonstrate volume regime detection."""
    print("\n4. Volume Regime Detection...")

    df = get_spy_data(start_date="2023-01-01", end_date="2023-06-30")
    detector = VolumeRegimeDetector(vol_window=20, zscore_threshold=1.0)

    # Detect volume regimes
    vol_regimes = detector.detect(df)

    print(f"   - Detected {(vol_regimes == 'high').sum()} high volume periods")
    print(f"   - Detected {(vol_regimes == 'normal').sum()} normal volume periods")
    print(f"   - Detected {(vol_regimes == 'low').sum()} low volume periods")

    # Volume statistics
    print(f"\n   - Volume statistics:")
    print(f"     - Min volume: {df['Volume'].min():,.0f}")
    print(f"     - Mean volume: {df['Volume'].mean():,.0f}")
    print(f"     - Max volume: {df['Volume'].max():,.0f}")
    print(f"     - Volume SMA-20: {df['Volume'].rolling(20).mean().iloc[-1]:,.0f}")

    return vol_regimes


def demo_composite_regime(df):
    """Demonstrate composite regime detection."""
    print("\n5. Composite Regime Detection...")

    detector = CompositeRegimeDetector()

    # Detect composite regimes
    composite_regimes = detector.detect(df)

    print(f"   - Composite regimes detected:")
    regime_counts = composite_regimes.value_counts()
    for regime, count in regime_counts.items():
        pct = 100 * count / len(composite_regimes)
        print(f"     - {regime:20s}: {count:3d} ({pct:5.1f}%)")

    # Show regime persistence
    print(f"\n   - Regime persistence (consecutive days):")
    regimes_shifted = composite_regimes != composite_regimes.shift()
    changes = regimes_shifted.sum()
    avg_persistence = len(composite_regimes) / changes if changes > 0 else 0
    print(f"     - Number of regime changes: {changes - 1}")
    print(f"     - Average regime duration: {avg_persistence:.1f} days")

    # Show recent regimes
    print(f"\n   - Recent composite regimes (last 10 days):")
    for i in range(min(10, len(composite_regimes))):
        idx = len(composite_regimes) - 10 + i
        date = df.index[idx]
        regime = composite_regimes.iloc[idx]
        close = df['Close'].iloc[idx]
        print(f"     {date.date()}: {regime:25s} - ${close:.2f}")

    return composite_regimes


def demo_regime_aware_strategy():
    """Demonstrate regime-aware strategy selection."""
    print("\n6. Regime-Aware Strategy Selection...")

    df = get_spy_data(start_date="2023-01-01", end_date="2023-06-30")

    # Detect regimes
    trend_detector = TrendRegimeDetector()
    trend_regimes = trend_detector.detect(df)

    vol_detector = VolatilityRegimeDetector()
    vol_regimes = vol_detector.detect(df)

    print("   - Strategy recommendations by regime:")

    regimes_df = pd.DataFrame({
        'trend': trend_regimes,
        'volatility': vol_regimes
    })

    combinations = regimes_df.value_counts().sort_values(ascending=False)

    print(f"\n   Regime Combinations & Recommended Strategies:")
    for (trend, vol), count in combinations.items():
        pct = 100 * count / len(regimes_df)

        # Suggest strategies
        if trend == 'trending' and vol == 'low':
            strategy = "Momentum / Trend-following"
        elif trend == 'trending' and vol == 'high':
            strategy = "Breakout / High-conviction"
        elif trend == 'ranging' and vol == 'low':
            strategy = "Mean reversion / Range trading"
        elif trend == 'ranging' and vol == 'high':
            strategy = "Options / Volatility strategies"
        else:
            strategy = "Mixed / Adaptive"

        print(f"   - {trend:10s} trend, {vol:4s} vol: {count:3d} days ({pct:5.1f}%) -> {strategy}")


def demo_detect_all_regimes():
    """Demonstrate detect_all_regimes function."""
    print("\n7. Detecting All Regimes (Complete Analysis)...")

    df = get_spy_data(start_date="2023-01-01", end_date="2023-06-30")

    # Detect all regimes
    all_regimes = detect_all_regimes(df)

    print(f"   - Detected regime columns: {list(all_regimes.columns)}")
    print(f"   - Number of rows: {len(all_regimes)}")

    # Show sample output
    print(f"\n   - Sample regime detection (first 5 rows):")
    print(all_regimes.head())

    # Summary statistics
    print(f"\n   - Summary by regime type:")
    for col in all_regimes.columns:
        if all_regimes[col].dtype == 'object':
            counts = all_regimes[col].value_counts()
            print(f"\n     {col}:")
            for val, count in counts.items():
                pct = 100 * count / len(all_regimes)
                print(f"       - {val:20s}: {count:3d} ({pct:5.1f}%)")

    return all_regimes


if __name__ == "__main__":
    # Run trend regime demo
    df, trend_regimes, trend_numeric = demo_trend_regime()

    # Run volatility regime demo
    vol_regimes = demo_volatility_regime()

    # Run volume regime demo
    volume_regimes = demo_volume_regime()

    # Run composite regime demo
    composite_regimes = demo_composite_regime(df)

    # Run regime-aware strategy demo
    demo_regime_aware_strategy()

    # Run detect all regimes demo
    all_regimes = demo_detect_all_regimes()

    print("\n" + "=" * 60)
    print("Phase 4 Demo Complete!")
    print("=" * 60)
    print("\nKey Insights:")
    print("- ADX detects trending vs ranging markets")
    print("- Volatility regimes identify option opportunities")
    print("- Volume regimes detect institutional activity")
    print("- Composite regimes combine multiple signals")
    print("- Different strategies perform better in different regimes")
    print("- Regime detection improves strategy selection")
    print("\nNext: Run phase5_demo.py for strategy predictions")
