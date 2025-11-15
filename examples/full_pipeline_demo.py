"""
Full Pipeline Demo: Complete End-to-End PRADO9 System

This demo integrates all 6 phases:
1. Phase 0: Configuration and State Management
2. Phase 1: CUSUM Filter and Bars
3. Phase 2: Triple Barrier Labeling
4. Phase 3: Feature Building
5. Phase 4: Regime Detection
6. Phase 5: Strategy Predictions
7. Phase 6: Model Training

This script demonstrates the complete flow from raw market data to
trained ensemble models and trading signals.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from afml_system.config import ConfigManager
from afml_system.state import BanditStateManager
from afml_system.data import (
    get_spy_data,
    cusum_filter,
    dollar_bars,
)
from afml_system.features import (
    build_feature_matrix,
    normalize_features,
)
from afml_system.labeling import (
    triple_barrier_labels,
    get_sample_weights,
)
from afml_system.regime import (
    TrendRegimeDetector,
    VolatilityRegimeDetector,
    detect_all_regimes,
)
from afml_system.strategies import (
    MomentumStrategy,
    MeanReversionStrategy,
    VolatilityStrategy,
)
from afml_system.models import PurgedKFold


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n  {title}")
    print("-" * 70)


def phase0_config_and_state():
    """Phase 0: Configuration and State Management."""
    print_section("PHASE 0: Configuration and State Management")

    print_subsection("1. Initializing Configuration Manager")
    config_mgr = ConfigManager()
    config = config_mgr.config

    print(f"     Data Configuration:")
    print(f"       - Symbols: {config.data.symbols}")
    print(f"       - Date range: {config.data.start_date} to {config.data.end_date}")
    print(f"       - Bar type: {config.data.bar_type}")

    print(f"\n     Model Configuration:")
    print(f"       - Estimators: {config.model.n_estimators}")
    print(f"       - Max depth: {config.model.max_depth}")
    print(f"       - CV folds: {config.model.cv_folds}")

    print_subsection("2. Initializing Bandit State Manager")
    state_mgr = BanditStateManager(state_file="/tmp/prado9_pipeline_demo.json")

    strategies = ["momentum", "mean_reversion", "volatility"]
    for strategy in strategies:
        state_mgr.register_arm(strategy)

    print(f"     Registered {len(strategies)} strategy arms")
    print(f"     Initial arm probabilities: {list(state_mgr.get_arm_probabilities().keys())}")

    return config_mgr, state_mgr


def phase1_data_generation(config_mgr):
    """Phase 1: CUSUM Filter and Information-Driven Bars."""
    print_section("PHASE 1: CUSUM Filter and Information-Driven Bars")

    print_subsection("1. Fetching Market Data")
    df = get_spy_data(
        start_date=config_mgr.config.data.start_date,
        end_date=config_mgr.config.data.end_date
    )
    print(f"     Loaded {len(df)} days of {config_mgr.config.data.symbols[0]} data")
    print(f"     Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")

    print_subsection("2. Generating CUSUM Events")
    events = cusum_filter(df['Close'], threshold=0.01)
    print(f"     Detected {len(events)} significant price movements")

    # Limit events for demo
    if len(events) > 30:
        events = events[:30]
        print(f"     Using first 30 events for demo")

    print_subsection("3. Generating Information-Driven Bars")
    from afml_system.data import get_optimal_bar_threshold
    dollar_threshold = get_optimal_bar_threshold(df, bar_type='dollar', target_bars=100)
    bars = dollar_bars(df, threshold=dollar_threshold)
    print(f"     Generated {len(bars)} dollar bars")

    return df, events


def phase2_labeling(df, events):
    """Phase 2: Triple Barrier Labeling."""
    print_section("PHASE 2: Triple Barrier Labeling")

    print_subsection("1. Calculating Daily Volatility")
    from afml_system.labeling import get_daily_volatility
    daily_vol = get_daily_volatility(df['Close'], span=100)
    print(f"     Volatility range: {daily_vol.min():.4f} - {daily_vol.max():.4f}")

    print_subsection("2. Generating Triple Barrier Labels")
    labels_df = triple_barrier_labels(
        close=df['Close'],
        events=events,
        pt_sl=[1.0, 1.0],
        min_ret=0.01,
        num_threads=1,
        vertical_barrier_days=5,
    )
    print(f"     Generated {len(labels_df)} labels")

    if 'label' in labels_df.columns:
        label_counts = labels_df['label'].value_counts()
        print(f"     Label distribution:")
        for label, count in label_counts.items():
            pct = 100 * count / len(labels_df)
            name = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}.get(label, 'Unknown')
            print(f"       - {name:10s}: {count} ({pct:.1f}%)")

    print_subsection("3. Calculating Sample Weights")
    sample_weights = get_sample_weights(labels_df)
    print(f"     Weight statistics:")
    print(f"       - Min: {sample_weights.min():.4f}")
    print(f"       - Mean: {sample_weights.mean():.4f}")
    print(f"       - Max: {sample_weights.max():.4f}")

    return labels_df, sample_weights


def phase3_features(df, labels_df):
    """Phase 3: Feature Building."""
    print_section("PHASE 3: Feature Building")

    print_subsection("1. Building Feature Matrix")
    features_df = build_feature_matrix(df)
    print(f"     Generated {len(features_df.columns)} features")

    print_subsection("2. Aligning with Labels")
    common_idx = features_df.index.intersection(labels_df.index)
    X = features_df.loc[common_idx].dropna()
    y = labels_df.loc[X.index, 'label'] if 'label' in labels_df.columns else pd.Series(0, index=X.index)

    print(f"     Final training set: {len(X)} samples")
    print(f"     Features: {X.shape[1]}")

    print_subsection("3. Normalizing Features")
    X_normalized = normalize_features(X, method='zscore')
    print(f"     Feature normalization complete")
    print(f"     Mean of means: {X_normalized.mean().mean():.8f}")
    print(f"     Mean of stds: {X_normalized.std().mean():.4f}")

    return X_normalized, y


def phase4_regimes(df):
    """Phase 4: Regime Detection."""
    print_section("PHASE 4: Regime Detection")

    print_subsection("1. Trend Regime Detection")
    trend_detector = TrendRegimeDetector()
    trend_regimes = trend_detector.detect(df)
    trending_days = (trend_regimes == 'trending').sum()
    ranging_days = (trend_regimes == 'ranging').sum()
    print(f"     Trending: {trending_days} days ({100*trending_days/len(df):.1f}%)")
    print(f"     Ranging: {ranging_days} days ({100*ranging_days/len(df):.1f}%)")

    print_subsection("2. Volatility Regime Detection")
    vol_detector = VolatilityRegimeDetector()
    vol_regimes = vol_detector.detect(df)
    high_vol_days = (vol_regimes == 'high').sum()
    low_vol_days = (vol_regimes == 'low').sum()
    print(f"     High volatility: {high_vol_days} days ({100*high_vol_days/len(df):.1f}%)")
    print(f"     Low volatility: {low_vol_days} days ({100*low_vol_days/len(df):.1f}%)")

    print_subsection("3. Regime Combinations")
    all_regimes = detect_all_regimes(df)
    print(f"     Detected {len(all_regimes.columns)} regime types")

    return trend_regimes, vol_regimes, all_regimes


def phase5_strategies(df):
    """Phase 5: Strategy Predictions."""
    print_section("PHASE 5: Strategy Predictions and Ensemble")

    print_subsection("1. Momentum Strategy")
    momentum = MomentumStrategy(lookback=20, threshold=0.02)
    momentum_signals = momentum.generate_signals(df)
    print(f"     Long signals: {(momentum_signals == 1).sum()}")
    print(f"     Short signals: {(momentum_signals == -1).sum()}")

    print_subsection("2. Mean Reversion Strategy")
    mean_rev = MeanReversionStrategy(lookback=20, zscore_threshold=2.0)
    mean_rev_signals = mean_rev.generate_signals(df)
    print(f"     Long signals: {(mean_rev_signals == 1).sum()}")
    print(f"     Short signals: {(mean_rev_signals == -1).sum()}")

    print_subsection("3. Volatility Strategy")
    vol_strategy = VolatilityStrategy(vol_window=20, vol_threshold=0.02)
    vol_signals = vol_strategy.generate_signals(df)
    print(f"     Long signals: {(vol_signals == 1).sum()}")
    print(f"     Short signals: {(vol_signals == -1).sum()}")

    print_subsection("4. Ensemble Signal")
    ensemble_signal = (momentum_signals + mean_rev_signals + vol_signals) / 3
    print(f"     Signal statistics:")
    print(f"       - Mean: {ensemble_signal.mean():.4f}")
    print(f"       - Std: {ensemble_signal.std():.4f}")
    print(f"       - Long bias: {ensemble_signal.mean() > 0}")

    return momentum_signals, mean_rev_signals, vol_signals, ensemble_signal


def phase6_training(X, y):
    """Phase 6: Model Training."""
    print_section("PHASE 6: Model Training")

    print_subsection("1. Cross-Validation Setup")
    pkf = PurgedKFold(n_splits=3)
    print(f"     Using {pkf.n_splits}-fold Purged K-Fold")

    print_subsection("2. Training Models")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    fold_results = []
    for fold_num, (train_idx, test_idx) in enumerate(pkf.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        fold_results.append({
            'fold': fold_num,
            'train_acc': train_acc,
            'test_acc': test_acc
        })
        print(f"     Fold {fold_num}: Train={train_acc:.4f}, Test={test_acc:.4f}")

    print_subsection("3. Model Performance")
    avg_train = np.mean([r['train_acc'] for r in fold_results])
    avg_test = np.mean([r['test_acc'] for r in fold_results])
    print(f"     Average Train Accuracy: {avg_train:.4f}")
    print(f"     Average Test Accuracy: {avg_test:.4f}")
    print(f"     Generalization Gap: {(avg_train - avg_test):.4f}")

    print_subsection("4. Feature Importance")
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X, y)

    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"     Top 5 Important Features:")
    for i, row in enumerate(importances.head(5).iterrows(), 1):
        feature, importance = row[1]
        print(f"       {i}. {feature[:25]:25s}: {importance:.6f}")

    return model, importances


def final_summary():
    """Print final summary and next steps."""
    print_section("PIPELINE COMPLETE")

    print_subsection("Summary of PRADO9 System")
    print(f"""
     Phase 0: Configuration and State Management
       - Loaded configuration
       - Initialized strategy bandit arms

     Phase 1: CUSUM Filter and Information-Driven Bars
       - Generated event triggers
       - Created information-driven bars

     Phase 2: Triple Barrier Labeling
       - Generated trading labels
       - Calculated sample weights

     Phase 3: Feature Building
       - Built 50+ technical features
       - Normalized features for training

     Phase 4: Regime Detection
       - Detected trending/ranging markets
       - Detected volatility regimes

     Phase 5: Strategy Predictions
       - Generated momentum signals
       - Generated mean reversion signals
       - Generated volatility signals
       - Created ensemble signals

     Phase 6: Model Training
       - Trained primary classifier
       - Evaluated with cross-validation
       - Analyzed feature importance
    """)

    print_subsection("Key Achievements")
    print(f"""
     - Integrated all 6 phases of the PRADO9 system
     - Demonstrated data pipeline from raw market data to ML models
     - Generated labels using triple barrier method
     - Built comprehensive feature matrix
     - Detected market regimes for adaptive strategies
     - Trained ensemble models with cross-validation
     - Analyzed feature importance
    """)

    print_subsection("Next Steps for Production")
    print(f"""
     1. Expand training data to multiple symbols and longer periods
     2. Fine-tune hyperparameters via grid/random search
     3. Implement walk-forward backtesting
     4. Add portfolio-level risk management
     5. Deploy models with real-time prediction capability
     6. Monitor model performance and retrain periodically
     7. Implement position sizing and execution logic
    """)

    print("\n" + "=" * 70)
    print("  PRADO9 Full Pipeline Demo Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Run all phases
    config_mgr, state_mgr = phase0_config_and_state()
    df, events = phase1_data_generation(config_mgr)
    labels_df, weights = phase2_labeling(df, events)
    X, y = phase3_features(df, labels_df)
    trend_regimes, vol_regimes, all_regimes = phase4_regimes(df)
    momentum_sig, mr_sig, vol_sig, ensemble_sig = phase5_strategies(df)
    model, importances = phase6_training(X, y)

    # Final summary
    final_summary()
