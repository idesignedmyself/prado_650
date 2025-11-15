"""
Phase 5 Demo: Strategy Predictions and Ensemble

This demo shows how to:
1. Generate predictions from multiple trading strategies
2. Aggregate strategy predictions
3. Filter predictions by market regime
4. Rank strategies by performance
5. Generate consensus signals
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from afml_system.data import get_spy_data
from afml_system.strategies import (
    MomentumStrategy,
    MeanReversionStrategy,
    VolatilityStrategy,
)
from afml_system.regime import TrendRegimeDetector, VolatilityRegimeDetector


def demo_individual_strategies():
    """Demonstrate individual strategy signals."""
    print("=" * 60)
    print("Phase 5: Strategy Predictions and Ensemble")
    print("=" * 60)

    print("\n1. Loading SPY data...")
    df = get_spy_data(start_date="2023-01-01", end_date="2023-06-30")
    print(f"   - Loaded {len(df)} days")

    # Momentum strategy
    print("\n2. Momentum Strategy Signals...")
    momentum = MomentumStrategy(lookback=20, threshold=0.02)
    momentum_signals = momentum.generate_signals(df)

    momentum_counts = momentum_signals.value_counts()
    print(f"   - Signal distribution:")
    for signal in sorted(momentum_counts.index):
        count = momentum_counts[signal]
        pct = 100 * count / len(momentum_signals)
        signal_name = {1: 'Long', 0: 'Neutral', -1: 'Short'}.get(signal, 'Unknown')
        print(f"     - {signal_name:10s}: {count:3d} ({pct:5.1f}%)")

    # Mean reversion strategy
    print("\n3. Mean Reversion Strategy Signals...")
    mean_rev = MeanReversionStrategy(lookback=20, zscore_threshold=2.0)
    mean_rev_signals = mean_rev.generate_signals(df)

    mr_counts = mean_rev_signals.value_counts()
    print(f"   - Signal distribution:")
    for signal in sorted(mr_counts.index):
        count = mr_counts[signal]
        pct = 100 * count / len(mean_rev_signals)
        signal_name = {1: 'Long', 0: 'Neutral', -1: 'Short'}.get(signal, 'Unknown')
        print(f"     - {signal_name:10s}: {count:3d} ({pct:5.1f}%)")

    # Volatility strategy
    print("\n4. Volatility Strategy Signals...")
    vol_strategy = VolatilityStrategy(vol_window=20, vol_threshold=0.02)
    vol_signals = vol_strategy.generate_signals(df)

    vol_counts = vol_signals.value_counts()
    print(f"   - Signal distribution:")
    for signal in sorted(vol_counts.index):
        count = vol_counts[signal]
        pct = 100 * count / len(vol_signals)
        signal_name = {1: 'Long', 0: 'Neutral', -1: 'Short'}.get(signal, 'Unknown')
        print(f"     - {signal_name:10s}: {count:3d} ({pct:5.1f}%)")

    return df, momentum_signals, mean_rev_signals, vol_signals


def demo_signal_agreement():
    """Demonstrate signal agreement between strategies."""
    print("\n5. Strategy Signal Agreement...")

    df = get_spy_data(start_date="2023-01-01", end_date="2023-06-30")

    momentum = MomentumStrategy(lookback=20, threshold=0.02)
    momentum_signals = momentum.generate_signals(df)

    mean_rev = MeanReversionStrategy(lookback=20, zscore_threshold=2.0)
    mean_rev_signals = mean_rev.generate_signals(df)

    vol_strategy = VolatilityStrategy(vol_window=20, vol_threshold=0.02)
    vol_signals = vol_strategy.generate_signals(df)

    # Combine signals
    combined = pd.DataFrame({
        'momentum': momentum_signals,
        'mean_reversion': mean_rev_signals,
        'volatility': vol_signals
    })

    # Calculate agreement
    all_long = (combined == 1).sum(axis=1) == 3
    all_short = (combined == -1).sum(axis=1) == 3
    all_neutral = (combined == 0).sum(axis=1) == 3
    majority_agree = (
        ((combined == 1).sum(axis=1) >= 2) |
        ((combined == -1).sum(axis=1) >= 2) |
        ((combined == 0).sum(axis=1) >= 2)
    )

    print(f"   - All strategies agree (long): {all_long.sum()} times ({100 * all_long.sum() / len(combined):.1f}%)")
    print(f"   - All strategies agree (short): {all_short.sum()} times ({100 * all_short.sum() / len(combined):.1f}%)")
    print(f"   - All strategies agree (neutral): {all_neutral.sum()} times ({100 * all_neutral.sum() / len(combined):.1f}%)")
    print(f"   - Majority agreement: {majority_agree.sum()} times ({100 * majority_agree.sum() / len(combined):.1f}%)")

    return combined


def demo_simple_ensemble():
    """Demonstrate simple ensemble averaging."""
    print("\n6. Simple Ensemble Averaging...")

    df = get_spy_data(start_date="2023-01-01", end_date="2023-06-30")

    momentum = MomentumStrategy(lookback=20, threshold=0.02)
    momentum_signals = momentum.generate_signals(df)

    mean_rev = MeanReversionStrategy(lookback=20, zscore_threshold=2.0)
    mean_rev_signals = mean_rev.generate_signals(df)

    vol_strategy = VolatilityStrategy(vol_window=20, vol_threshold=0.02)
    vol_signals = vol_strategy.generate_signals(df)

    # Average signals
    ensemble_signal = (momentum_signals + mean_rev_signals + vol_signals) / 3

    print(f"   - Ensemble signal statistics:")
    print(f"     - Mean: {ensemble_signal.mean():.4f}")
    print(f"     - Std: {ensemble_signal.std():.4f}")
    print(f"     - Min: {ensemble_signal.min():.4f}")
    print(f"     - Max: {ensemble_signal.max():.4f}")

    # Threshold ensemble signal
    long_signals = (ensemble_signal > 0.33).sum()
    short_signals = (ensemble_signal < -0.33).sum()
    neutral_signals = len(ensemble_signal) - long_signals - short_signals

    print(f"\n   - Ensemble signal distribution (threshold 0.33):")
    print(f"     - Long: {long_signals} ({100 * long_signals / len(ensemble_signal):.1f}%)")
    print(f"     - Short: {short_signals} ({100 * short_signals / len(ensemble_signal):.1f}%)")
    print(f"     - Neutral: {neutral_signals} ({100 * neutral_signals / len(ensemble_signal):.1f}%)")

    return ensemble_signal


def demo_regime_filtered_signals():
    """Demonstrate regime-based signal filtering."""
    print("\n7. Regime-Filtered Strategy Signals...")

    df = get_spy_data(start_date="2023-01-01", end_date="2023-06-30")

    # Get regime
    trend_detector = TrendRegimeDetector()
    regimes = trend_detector.detect(df)

    # Get signals
    momentum = MomentumStrategy(lookback=20, threshold=0.02)
    momentum_signals = momentum.generate_signals(df)

    # Filter by regime
    trending_signals = momentum_signals[regimes == 'trending']
    ranging_signals = momentum_signals[regimes == 'ranging']

    print(f"   - Momentum strategy performance by regime:")
    print(f"     - Trending regime: {len(trending_signals)} days")
    print(f"       - Long signals: {(trending_signals == 1).sum()} ({100 * (trending_signals == 1).sum() / len(trending_signals):.1f}%)")
    print(f"       - Short signals: {(trending_signals == -1).sum()} ({100 * (trending_signals == -1).sum() / len(trending_signals):.1f}%)")

    print(f"     - Ranging regime: {len(ranging_signals)} days")
    print(f"       - Long signals: {(ranging_signals == 1).sum()} ({100 * (ranging_signals == 1).sum() / len(ranging_signals):.1f}%)")
    print(f"       - Short signals: {(ranging_signals == -1).sum()} ({100 * (ranging_signals == -1).sum() / len(ranging_signals):.1f}%)")

    # Strategy recommendation
    print(f"\n   - Strategy Recommendations:")
    print(f"     - Momentum works better in: {'TRENDING' if (trending_signals == 1).sum() > (ranging_signals == 1).sum() else 'RANGING'} markets")


def demo_signal_quality():
    """Demonstrate signal quality metrics."""
    print("\n8. Signal Quality Metrics...")

    df = get_spy_data(start_date="2023-01-01", end_date="2023-06-30")

    # Get signals and returns
    momentum = MomentumStrategy(lookback=20, threshold=0.02)
    signals = momentum.generate_signals(df)
    returns = df['Close'].pct_change()

    # Calculate signal-return correlation
    corr = signals.corr(returns)
    print(f"   - Signal correlation with next returns: {corr:.4f}")

    # Calculate hit rate (positive returns after buy signal)
    long_indices = signals[signals == 1].index
    long_returns = returns[returns.index.isin(long_indices)]
    hit_rate = (long_returns > 0).sum() / len(long_returns) if len(long_returns) > 0 else 0

    print(f"   - Hit rate (long signals): {hit_rate:.2%}")

    # Win/loss ratio
    long_wins = (long_returns > 0).sum()
    long_losses = (long_returns < 0).sum()

    if long_wins > 0 and long_losses > 0:
        avg_win = long_returns[long_returns > 0].mean()
        avg_loss = abs(long_returns[long_returns < 0].mean())
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        print(f"   - Win/Loss Statistics:")
        print(f"     - Wins: {long_wins}")
        print(f"     - Losses: {long_losses}")
        print(f"     - Avg win: {avg_win:.4f}")
        print(f"     - Avg loss: {avg_loss:.4f}")
        print(f"     - Win/Loss ratio: {win_loss_ratio:.4f}")


if __name__ == "__main__":
    # Run individual strategies
    df, momentum_signals, mean_rev_signals, vol_signals = demo_individual_strategies()

    # Run signal agreement
    combined = demo_signal_agreement()

    # Run ensemble
    ensemble = demo_simple_ensemble()

    # Run regime filtering
    demo_regime_filtered_signals()

    # Run signal quality
    demo_signal_quality()

    print("\n" + "=" * 60)
    print("Phase 5 Demo Complete!")
    print("=" * 60)
    print("\nKey Insights:")
    print("- Multiple strategies capture different market inefficiencies")
    print("- Ensemble methods improve signal robustness")
    print("- Strategy agreement increases signal confidence")
    print("- Regime filtering optimizes strategy selection")
    print("- Win/loss ratios evaluate strategy quality")
    print("\nNext: Run phase6_demo.py for model training")
