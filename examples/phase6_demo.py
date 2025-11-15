"""
Phase 6 Demo: Model Training

This demo shows how to:
1. Prepare training data with labels
2. Train a primary classification model
3. Train a meta-model for strategy selection
4. Evaluate model performance
5. Save and load trained models
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
from afml_system.features import (
    build_feature_matrix,
    normalize_features,
)
from afml_system.labeling import (
    triple_barrier_labels,
    get_daily_volatility,
    get_sample_weights,
)
from afml_system.models import (
    train_primary_model,
    train_meta_model,
    ModelPersistence,
    PurgedKFold,
)


def demo_prepare_training_data():
    """Demonstrate training data preparation."""
    print("=" * 60)
    print("Phase 6: Model Training")
    print("=" * 60)

    print("\n1. Preparing Training Data...")

    print("   a) Loading market data...")
    df = get_spy_data(start_date="2023-01-01", end_date="2023-06-30")
    close_prices = df['Close']
    print(f"      - {len(df)} days of data")

    print("   b) Generating events...")
    events = cusum_filter(close_prices, threshold=0.01)
    print(f"      - {len(events)} events detected")

    # Limit events for faster training
    if len(events) > 30:
        events = events[:30]
        print(f"      - Using first 30 events for demo")

    print("   c) Generating labels...")
    labels_df = triple_barrier_labels(
        close=close_prices,
        events=events,
        pt_sl=[1.0, 1.0],
        min_ret=0.01,
        num_threads=1,
        vertical_barrier_days=5,
    )
    print(f"      - {len(labels_df)} labels generated")

    print("   d) Building features...")
    features_df = build_feature_matrix(df)
    print(f"      - {len(features_df.columns)} features created")

    print("   e) Cleaning data...")
    # Align features with labels
    common_idx = features_df.index.intersection(labels_df.index)
    X = features_df.loc[common_idx].dropna()
    y = labels_df.loc[X.index, 'label'] if 'label' in labels_df.columns else pd.Series(0, index=X.index)

    print(f"      - Final training set: {len(X)} samples")
    print(f"      - Features: {X.shape[1]}")
    print(f"      - Labels distribution:")
    label_counts = y.value_counts()
    for label, count in label_counts.items():
        pct = 100 * count / len(y)
        print(f"        - Label {label}: {count} ({pct:.1f}%)")

    print("   f) Normalizing features...")
    X_normalized = normalize_features(X, method='zscore')
    print(f"      - Feature normalization complete")

    print("   g) Calculating sample weights...")
    sample_weights = get_sample_weights(labels_df.loc[X.index])
    print(f"      - Weights calculated for {len(sample_weights)} samples")
    print(f"      - Weight range: {sample_weights.min():.4f} to {sample_weights.max():.4f}")

    return X_normalized, y, sample_weights, labels_df.loc[X.index]


def demo_model_training(X, y, sample_weights):
    """Demonstrate primary model training."""
    print("\n2. Training Primary Model...")

    print("   a) Initializing model parameters...")
    model_params = {
        'n_estimators': 50,
        'max_depth': 5,
        'min_samples_leaf': 10,
        'random_state': 42,
        'n_jobs': -1
    }
    print(f"      - Model type: Random Forest")
    print(f"      - Estimators: {model_params['n_estimators']}")
    print(f"      - Max depth: {model_params['max_depth']}")

    print("   b) Training on full dataset...")
    try:
        model, metrics = train_primary_model(
            X=X,
            y=y,
            sample_weight=sample_weights,
            model_type='rf',
            model_params=model_params,
            cv_folds=3
        )
        print(f"      - Model trained successfully")
        print(f"      - Training metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"        - {key}: {value:.4f}")
            else:
                print(f"        - {key}: {value}")

        return model

    except Exception as e:
        print(f"      - Note: {e}")
        print(f"      - Falling back to simple model")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**model_params)
        model.fit(X, y, sample_weight=sample_weights)
        return model


def demo_cross_validation(X, y):
    """Demonstrate cross-validation."""
    print("\n3. Cross-Validation...")

    print("   a) Setting up Purged K-Fold...")
    pkf = PurgedKFold(n_splits=3)

    fold_num = 0
    fold_results = []

    print(f"   b) Training on {pkf.n_splits} folds...")
    for train_idx, test_idx in pkf.split(X):
        fold_num += 1
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        print(f"\n      Fold {fold_num}:")
        print(f"      - Train size: {len(X_train)}")
        print(f"      - Test size: {len(X_test)}")

        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, precision_score, recall_score

            model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            model.fit(X_train, y_train)

            train_acc = accuracy_score(y_train, model.predict(X_train))
            test_acc = accuracy_score(y_test, model.predict(X_test))

            print(f"      - Train accuracy: {train_acc:.4f}")
            print(f"      - Test accuracy: {test_acc:.4f}")

            fold_results.append({'fold': fold_num, 'train_acc': train_acc, 'test_acc': test_acc})

        except Exception as e:
            print(f"      - Error in fold: {e}")

    if fold_results:
        avg_train = np.mean([r['train_acc'] for r in fold_results])
        avg_test = np.mean([r['test_acc'] for r in fold_results])
        print(f"\n      Average train accuracy: {avg_train:.4f}")
        print(f"      Average test accuracy: {avg_test:.4f}")


def demo_feature_importance(model, X):
    """Demonstrate feature importance."""
    print("\n4. Feature Importance Analysis...")

    try:
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print(f"   - Top 10 important features:")
        for i, row in enumerate(feature_importance_df.head(10).iterrows(), 1):
            feature, importance = row[1]
            print(f"     {i:2d}. {feature:30s}: {importance:.6f}")

        # Feature importance distribution
        print(f"\n   - Feature importance distribution:")
        print(f"     - Min: {importances.min():.6f}")
        print(f"     - Mean: {importances.mean():.6f}")
        print(f"     - Max: {importances.max():.6f}")
        print(f"     - Non-zero features: {(importances > 0).sum()} / {len(importances)}")

    except Exception as e:
        print(f"   - Note: {e}")


def demo_model_persistence(model, X):
    """Demonstrate model saving and loading."""
    print("\n5. Model Persistence...")

    print("   a) Saving model...")
    persistence = ModelPersistence()

    model_path = "/tmp/prado9_model_demo.pkl"
    try:
        persistence.save_model(model, model_path)
        print(f"      - Model saved to: {model_path}")

        print("   b) Loading model...")
        loaded_model = persistence.load_model(model_path)
        print(f"      - Model loaded successfully")

        print("   c) Testing loaded model...")
        original_pred = model.predict(X.iloc[:5])
        loaded_pred = loaded_model.predict(X.iloc[:5])

        match = np.array_equal(original_pred, loaded_pred)
        print(f"      - Predictions match: {match}")

    except Exception as e:
        print(f"      - Note: {e}")


def demo_meta_model(X, y):
    """Demonstrate meta-model training."""
    print("\n6. Training Meta-Model (Strategy Selector)...")

    # Create synthetic strategy predictions
    np.random.seed(42)
    strategy_preds = pd.DataFrame({
        'momentum': np.random.choice([-1, 0, 1], len(X)),
        'mean_reversion': np.random.choice([-1, 0, 1], len(X)),
        'volatility': np.random.choice([-1, 0, 1], len(X)),
    }, index=X.index)

    print(f"   a) Strategy predictions shape: {strategy_preds.shape}")
    print(f"   b) Strategy correlation with labels:")

    for strategy in strategy_preds.columns:
        corr = strategy_preds[strategy].corr(y)
        print(f"      - {strategy}: {corr:.4f}")

    try:
        print(f"   c) Training meta-model...")
        meta_model, meta_metrics = train_meta_model(
            X=strategy_preds,
            y=y,
            model_type='rf',
            cv_folds=3
        )
        print(f"      - Meta-model trained successfully")

    except Exception as e:
        print(f"      - Note: {e}")


if __name__ == "__main__":
    # Prepare training data
    X, y, weights, labels_df = demo_prepare_training_data()

    # Train primary model
    model = demo_model_training(X, y, weights)

    # Run cross-validation
    demo_cross_validation(X, y)

    # Feature importance
    demo_feature_importance(model, X)

    # Model persistence
    demo_model_persistence(model, X)

    # Meta-model
    demo_meta_model(X, y)

    print("\n" + "=" * 60)
    print("Phase 6 Demo Complete!")
    print("=" * 60)
    print("\nKey Insights:")
    print("- Proper data preparation is critical for model success")
    print("- Cross-validation prevents overfitting")
    print("- Feature importance guides feature engineering")
    print("- Model persistence enables production deployment")
    print("- Meta-models optimize strategy selection")
    print("\nNext: Run full_pipeline_demo.py for end-to-end example")
