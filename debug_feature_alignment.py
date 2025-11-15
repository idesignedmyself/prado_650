#!/usr/bin/env python3
"""
Feature Alignment Diagnostic Sweep
Identifies where samples drop in the feature-building pipeline
"""
import pandas as pd
import numpy as np
from src.afml_system.data.fetch import prepare_training_data
from src.afml_system.features.feature_union import build_feature_matrix
from src.afml_system.labeling.triple_barrier import triple_barrier_labels
from src.afml_system.regime.detection import detect_all_regimes

def audit_feature_alignment(symbol="QQQ", start_date="2023-01-01", end_date="2023-06-30"):
    """Run complete diagnostic sweep"""

    print("=" * 80)
    print("FEATURE ALIGNMENT DIAGNOSTIC SWEEP")
    print("=" * 80)
    print(f"\nSymbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")

    # Stage 1: Raw Data
    print("\n" + "=" * 80)
    print("[STAGE 1] Fetching Raw Data")
    print("=" * 80)

    data = prepare_training_data(symbol, start_date, end_date)

    print(f"✓ Raw Data Shape: {data.shape}")
    print(f"✓ Columns: {data.columns.tolist()}")
    print(f"✓ Index Type: {type(data.index)}")
    print(f"✓ Index Range: {data.index[0]} to {data.index[-1]}")
    print(f"✓ NaN Count: {data.isna().sum().sum()}")
    print(f"\nFirst 3 rows:")
    print(data.head(3))

    stage1_rows = len(data)

    # Stage 2: Build Features
    print("\n" + "=" * 80)
    print("[STAGE 2] Building Features")
    print("=" * 80)

    try:
        features = build_feature_matrix(data)
        print(f"✓ Features Shape: {features.shape}")
        print(f"✓ Features Columns ({len(features.columns)}): {features.columns.tolist()}")
        print(f"✓ Features Index Type: {type(features.index)}")
        if len(features) > 0:
            print(f"✓ Features Index Range: {features.index[0]} to {features.index[-1]}")
            print(f"✓ Features NaN Count: {features.isna().sum().sum()}")
            print(f"\nFirst 3 feature rows:")
            print(features.head(3))
        else:
            print("⚠️ WARNING: Features DataFrame is EMPTY!")
            print(f"   Expected index: {data.index[:3].tolist()}")
            print(f"   Actual index: {features.index.tolist()}")
    except Exception as e:
        print(f"❌ ERROR building features: {e}")
        import traceback
        traceback.print_exc()
        features = pd.DataFrame()

    stage2_rows = len(features)

    # Stage 3: Generate Labels
    print("\n" + "=" * 80)
    print("[STAGE 3] Generating Labels")
    print("=" * 80)

    try:
        events = data.index
        labels = triple_barrier_labels(
            data['Close'],
            events,
            pt_sl=[1.0, 1.0],
            vertical_barrier_days=5
        )
        print(f"✓ Labels Shape: {labels.shape}")
        print(f"✓ Labels Columns: {labels.columns.tolist()}")
        print(f"✓ Labels Index Type: {type(labels.index)}")
        if len(labels) > 0:
            print(f"✓ Labels Index Range: {labels.index[0]} to {labels.index[-1]}")
            print(f"✓ Labels NaN Count: {labels.isna().sum().sum()}")
            print(f"\nFirst 3 label rows:")
            print(labels.head(3))
        else:
            print("⚠️ WARNING: Labels DataFrame is EMPTY!")
    except Exception as e:
        print(f"❌ ERROR generating labels: {e}")
        import traceback
        traceback.print_exc()
        labels = pd.DataFrame()

    stage3_rows = len(labels)

    # Stage 4: Merge Features + Labels
    print("\n" + "=" * 80)
    print("[STAGE 4] Merging Features + Labels")
    print("=" * 80)

    if len(features) > 0 and len(labels) > 0:
        # Check index overlap
        common_idx = features.index.intersection(labels.index)
        print(f"✓ Common Index Size: {len(common_idx)}")
        print(f"✓ Features Index Size: {len(features.index)}")
        print(f"✓ Labels Index Size: {len(labels.index)}")

        if len(common_idx) > 0:
            merged = features.loc[common_idx].join(labels.loc[common_idx], how='inner')
            print(f"✓ Merged Shape: {merged.shape}")
            print(f"✓ Merged NaN Count: {merged.isna().sum().sum()}")
            print(f"\nNaN Summary (top 10 columns):")
            nan_summary = merged.isna().sum().sort_values(ascending=False).head(10)
            print(nan_summary)
        else:
            print("❌ ZERO COMMON INDEX between features and labels!")
            print(f"\nFeatures index sample: {features.index[:5].tolist()}")
            print(f"Labels index sample: {labels.index[:5].tolist()}")
            merged = pd.DataFrame()
    else:
        print("❌ Cannot merge - features or labels are empty!")
        merged = pd.DataFrame()

    stage4_rows = len(merged) if 'merged' in locals() else 0

    # Summary Table
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)

    print("\n| Checkpoint          | Rows Before | Rows After | Δ %      | NaN Cols | Notes")
    print("|---------------------|-------------|------------|----------|----------|----------------------")

    stage2_delta = ((stage2_rows - stage1_rows) / stage1_rows * 100) if stage1_rows > 0 else 0
    print(f"| Pre-feature         | {stage1_rows:11} | {stage1_rows:10} | {0:7.1f}% | {0:8} | ✅ Clean data")

    stage3_delta = ((stage2_rows - stage1_rows) / stage1_rows * 100) if stage1_rows > 0 else 0
    status2 = "✅ Normal" if stage2_rows > 0 else "❌ EMPTY FEATURES"
    print(f"| Feature build       | {stage1_rows:11} | {stage2_rows:10} | {stage3_delta:7.1f}% | {0:8} | {status2}")

    stage4_delta = ((stage4_rows - stage2_rows) / stage2_rows * 100) if stage2_rows > 0 else -100
    status4 = "✅ Normal" if stage4_rows > stage2_rows * 0.95 else "⚠️ Index misalignment"
    print(f"| Label merge         | {stage2_rows:11} | {stage4_rows:10} | {stage4_delta:7.1f}% | {0:8} | {status4}")

    # Diagnosis
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    if stage2_rows == 0:
        print("\n❌ PROBLEM: Features DataFrame is EMPTY (0 rows)")
        print("\nLikely Causes:")
        print("  1. Column name mismatch (features expect lowercase, data has capitalized)")
        print("  2. Index alignment issue in build_feature_matrix()")
        print("  3. All features have NaN and are being dropped")
        print("\nRecommended Fix:")
        print("  - Check build_feature_matrix() in src/afml_system/features/feature_union.py")
        print("  - Ensure data column names match what feature functions expect")
        print("  - Add debug prints to see where rows are dropped")

    elif stage4_rows == 0:
        print("\n❌ PROBLEM: Merge produces ZERO rows")
        print("\nLikely Causes:")
        print("  1. Features and labels have non-overlapping indices")
        print("  2. Label index is shifted by 1 (forward-looking bias prevention)")
        print("\nRecommended Fix:")
        print("  - Align label index to match feature index")
        print("  - Use: labels.index = features.index[:len(labels)]")

    elif stage4_rows < stage2_rows * 0.95:
        print(f"\n⚠️ WARNING: Merge lost {stage2_rows - stage4_rows} rows ({100 - stage4_delta:.1f}%)")
        print("\nLikely Causes:")
        print("  1. Partial index overlap")
        print("  2. NaN values in merged data")

    else:
        print("\n✅ SUCCESS: Feature alignment is working correctly!")
        print(f"   - {stage4_rows} samples ready for training")

    print("\n" + "=" * 80)
    print("END OF DIAGNOSTIC SWEEP")
    print("=" * 80)

if __name__ == "__main__":
    audit_feature_alignment()
