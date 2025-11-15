"""
Phase 3 Demo: Feature Building

This demo shows how to:
1. Build a feature matrix from market data
2. Use technical indicators as features
3. Normalize features for model training
4. Select top features by importance
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from afml_system.data import get_spy_data
from afml_system.features import (
    build_feature_matrix,
    build_extended_features,
    select_top_features,
    normalize_features,
    sma,
    ema,
    rsi,
    macd,
)


def demo_basic_features():
    """Demonstrate basic feature building."""
    print("=" * 60)
    print("Phase 3: Feature Building")
    print("=" * 60)

    print("\n1. Loading SPY data...")
    df = get_spy_data(start_date="2023-01-01", end_date="2023-06-30")
    print(f"   - Loaded {len(df)} days of data")
    print(f"   - Columns: {list(df.columns)}")

    # Manual technical indicators
    print("\n2. Building Technical Indicators Manually...")

    # Simple Moving Average
    print("   - Calculating SMAs...")
    sma_5 = sma(df['Close'], window=5)
    sma_20 = sma(df['Close'], window=20)
    sma_50 = sma(df['Close'], window=50)
    print(f"     - SMA-5: {sma_5.iloc[-1]:.2f}")
    print(f"     - SMA-20: {sma_20.iloc[-1]:.2f}")
    print(f"     - SMA-50: {sma_50.iloc[-1]:.2f}")

    # Exponential Moving Average
    print("   - Calculating EMAs...")
    ema_12 = ema(df['Close'], span=12)
    ema_26 = ema(df['Close'], span=26)
    print(f"     - EMA-12: {ema_12.iloc[-1]:.2f}")
    print(f"     - EMA-26: {ema_26.iloc[-1]:.2f}")

    # RSI
    print("   - Calculating RSI...")
    rsi_14 = rsi(df['Close'], window=14)
    print(f"     - RSI-14: {rsi_14.iloc[-1]:.2f}")

    # MACD
    print("   - Calculating MACD...")
    macd_result = macd(df['Close'])
    if 'MACD' in macd_result.columns:
        print(f"     - MACD line: {macd_result['MACD'].iloc[-1]:.6f}")
        print(f"     - Signal line: {macd_result['Signal'].iloc[-1]:.6f}")
        print(f"     - Histogram: {macd_result['Histogram'].iloc[-1]:.6f}")

    # Price-based features
    print("   - Creating price-based features...")
    returns = df['Close'].pct_change()
    log_returns = np.log(df['Close'] / df['Close'].shift(1))
    high_low_ratio = df['High'] / df['Low']
    close_open_ratio = df['Close'] / df['Open']

    print(f"     - Returns (mean): {returns.mean():.6f}")
    print(f"     - Log Returns (mean): {log_returns.mean():.6f}")
    print(f"     - High/Low ratio (mean): {high_low_ratio.mean():.6f}")
    print(f"     - Close/Open ratio (mean): {close_open_ratio.mean():.6f}")

    # Volume-based features
    print("   - Creating volume-based features...")
    vol_sma = sma(df['Volume'], window=20)
    volume_ratio = df['Volume'] / vol_sma
    print(f"     - Volume SMA-20: {vol_sma.iloc[-1]:,.0f}")
    print(f"     - Volume Ratio: {volume_ratio.iloc[-1]:.4f}")

    return df


def demo_feature_matrix():
    """Demonstrate automatic feature matrix building."""
    print("\n3. Building Feature Matrix Automatically...")

    df = get_spy_data(start_date="2023-01-01", end_date="2023-06-30")

    # Build feature matrix
    print("   - Calling build_feature_matrix()...")
    features_df = build_feature_matrix(df)

    print(f"   - Generated {len(features_df)} rows")
    print(f"   - Generated {len(features_df.columns)} features:")
    for i, col in enumerate(features_df.columns[:20], 1):
        print(f"     {i:2d}. {col}")
    if len(features_df.columns) > 20:
        print(f"     ... and {len(features_df.columns) - 20} more features")

    # Feature statistics
    print("\n   - Feature statistics (sample):")
    print(f"     - Non-null values: {features_df.notna().sum().min()}")
    print(f"     - Missing values: {features_df.isna().sum().max()}")

    return features_df


def demo_extended_features(df):
    """Demonstrate extended feature building."""
    print("\n4. Building Extended Features...")

    # Build extended features
    print("   - Calling build_extended_features()...")
    ext_features_df = build_extended_features(df)

    print(f"   - Generated {len(ext_features_df)} rows")
    print(f"   - Generated {len(ext_features_df.columns)} extended features:")

    # Show columns by category
    feature_categories = {}
    for col in ext_features_df.columns:
        category = col.split('_')[0]
        if category not in feature_categories:
            feature_categories[category] = []
        feature_categories[category].append(col)

    for category, cols in sorted(feature_categories.items()):
        print(f"     - {category}: {len(cols)} features")

    return ext_features_df


def demo_feature_selection(features_df):
    """Demonstrate feature selection."""
    print("\n5. Feature Selection...")

    # Create dummy labels for demo
    np.random.seed(42)
    labels = pd.Series(
        np.random.choice([-1, 0, 1], size=len(features_df)),
        index=features_df.index
    )

    # Select top features
    print("   - Selecting top 10 features using feature importance...")
    top_features_df = select_top_features(features_df, labels, n_features=10)

    print(f"   - Selected {len(top_features_df.columns)} features:")
    for i, col in enumerate(top_features_df.columns, 1):
        print(f"     {i:2d}. {col}")

    return top_features_df


def demo_feature_normalization(features_df):
    """Demonstrate feature normalization."""
    print("\n6. Feature Normalization...")

    # Show original statistics
    print("   - Original feature statistics:")
    print(f"     - Feature means range: {features_df.mean().min():.6f} to {features_df.mean().max():.6f}")
    print(f"     - Feature stds range: {features_df.std().min():.6f} to {features_df.std().max():.6f}")

    # Normalize using z-score
    print("   - Normalizing with z-score method...")
    normalized_df = normalize_features(features_df, method='zscore')

    print("   - Normalized feature statistics:")
    print(f"     - Feature means: {normalized_df.mean().mean():.10f} (should be ~0)")
    print(f"     - Feature stds: {normalized_df.std().mean():.6f} (should be ~1)")
    print(f"     - Min normalized value: {normalized_df.values.min():.4f}")
    print(f"     - Max normalized value: {normalized_df.values.max():.4f}")

    # Normalize using min-max
    print("   - Normalizing with min-max method...")
    minmax_df = normalize_features(features_df, method='minmax')

    print("   - Min-max normalized statistics:")
    print(f"     - Min value: {minmax_df.values.min():.6f}")
    print(f"     - Max value: {minmax_df.values.max():.6f}")
    print(f"     - Mean value: {minmax_df.values.mean():.6f}")

    return normalized_df, minmax_df


def demo_feature_engineering_pipeline():
    """Show complete feature engineering pipeline."""
    print("\n7. Complete Feature Engineering Pipeline:")

    df = get_spy_data(start_date="2023-01-01", end_date="2023-06-30")

    print("   Step 1: Load market data")
    print("   Step 2: Build feature matrix")
    features = build_feature_matrix(df)

    print("   Step 3: Remove rows with NaN values")
    features_clean = features.dropna()
    print(f"        - Cleaned: {len(features)} -> {len(features_clean)} rows")

    print("   Step 4: Normalize features")
    features_norm = normalize_features(features_clean, method='zscore')

    print("   Step 5: Ready for model training")
    print(f"        - Shape: {features_norm.shape}")
    print(f"        - Data type: {features_norm.values.dtype}")

    return features_norm


if __name__ == "__main__":
    # Run basic features demo
    df = demo_basic_features()

    # Run feature matrix demo
    features_df = demo_feature_matrix()

    # Run extended features demo
    ext_features = demo_extended_features(df)

    # Run feature selection demo
    top_features = demo_feature_selection(features_df)

    # Run normalization demo
    normalized, minmax = demo_feature_normalization(features_df)

    # Run complete pipeline
    pipeline_features = demo_feature_engineering_pipeline()

    print("\n" + "=" * 60)
    print("Phase 3 Demo Complete!")
    print("=" * 60)
    print("\nKey Insights:")
    print("- Technical indicators provide predictive signals")
    print("- Multiple features improve model performance")
    print("- Feature normalization is critical for tree-based models")
    print("- Feature selection reduces dimensionality and noise")
    print("- Z-score normalization centers features around 0")
    print("- Min-max normalization bounds features to [0, 1]")
    print("\nNext: Run phase4_demo.py for regime detection")
