"""
Main PRADO9 pipeline.
Orchestrates train_ensemble, predict_ensemble, backtest_comprehensive.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from .data.fetch import fetch_ohlcv, prepare_training_data
from .features.feature_union import build_feature_matrix
from .labeling.triple_barrier import triple_barrier_labels, get_daily_volatility
from .labeling.weights import get_sample_weights
from .regime.detection import detect_all_regimes
from .models.trainer import train_primary_model, train_meta_model, train_strategy_models
from .models.persistence import ModelPersistence
from .strategies.ensemble import run_all_strategies, aggregate_strategy_predictions
from .allocation.hybrid_allocator import HybridAllocator
from .evaluation.backtest import simple_backtest, walk_forward_backtest
from .evaluation.metrics import get_all_metrics


def train_ensemble(
    symbols: List[str],
    start_date: str,
    end_date: str,
    config: Optional[Dict] = None,
    save_models: bool = True,
    models_dir: str = "models"
) -> Dict[str, Any]:
    """
    Train complete PRADO9 ensemble.

    Pipeline:
    1. Fetch data
    2. Build features
    3. Generate labels
    4. Detect regimes
    5. Train primary model
    6. Train meta-model
    7. Train strategy models
    8. Save models

    Args:
        symbols: List of symbols to train on
        start_date: Start date
        end_date: End date
        config: Configuration dict
        save_models: Whether to save models
        models_dir: Directory for models

    Returns:
        Dictionary with trained models and metrics
    """
    print("=" * 60)
    print("PRADO9 ENSEMBLE TRAINING")
    print("=" * 60)

    # Default config
    if config is None:
        config = {
            'pt_sl': [1.0, 1.0],
            'vertical_barrier_days': 5,
            'model_type': 'rf',
            'cv_folds': 5
        }

    results = {}

    # Step 1: Fetch data
    print("\n[1/8] Fetching market data...")
    all_data = {}
    for symbol in symbols:
        df = prepare_training_data(symbol, start_date, end_date)
        all_data[symbol] = df
        print(f"  {symbol}: {len(df)} bars")

    # Use first symbol as primary
    primary_symbol = symbols[0]
    df = all_data[primary_symbol]

    # Step 2: Build features
    print("\n[2/8] Building features...")
    features = build_feature_matrix(df)
    print(f"  Features shape: {features.shape}")

    # Step 3: Generate labels
    print("\n[3/8] Generating labels...")
    events = df.index
    labels = triple_barrier_labels(
        df['Close'],
        events,
        pt_sl=config['pt_sl'],
        vertical_barrier_days=config['vertical_barrier_days']
    )
    print(f"  Labels generated: {len(labels)}")

    # Step 4: Detect regimes
    print("\n[4/8] Detecting regimes...")
    regimes = detect_all_regimes(df)
    print(f"  Regimes detected: {regimes.shape}")

    # Step 5: Train primary model
    print("\n[5/8] Training primary model...")

    # Align data
    idx = features.index.intersection(labels.index)
    X = features.loc[idx]
    y = labels.loc[idx, 'label']

    # Get sample weights
    sample_weights = get_sample_weights(labels, df['Close'], method='return')

    primary_model, primary_metrics = train_primary_model(
        X, y,
        sample_weight=sample_weights,
        samples_info_sets=labels['t1'],
        model_type=config['model_type'],
        cv_folds=config['cv_folds']
    )

    print(f"  Primary model CV score: {primary_metrics['cv_mean']:.4f} +/- {primary_metrics['cv_std']:.4f}")
    results['primary_model'] = primary_model
    results['primary_metrics'] = primary_metrics

    # Step 6: Train meta-model
    print("\n[6/8] Training meta-model...")

    # Get primary predictions
    primary_pred = pd.Series(primary_model.predict(X), index=X.index)

    # Create meta-features
    X_meta = X.copy()
    X_meta['primary_pred'] = primary_pred

    # Meta-labels (whether prediction was correct)
    y_meta = ((primary_pred * y) > 0).astype(int)

    meta_model, meta_metrics = train_meta_model(X_meta, y_meta)

    print(f"  Meta-model accuracy: {meta_metrics['accuracy']:.4f}")
    results['meta_model'] = meta_model
    results['meta_metrics'] = meta_metrics

    # Step 7: Train strategy models
    print("\n[7/8] Training strategy models...")

    # For demonstration, create simple strategy-specific features
    from .strategies.momentum import MomentumStrategy
    from .strategies.mean_reversion import MeanReversionStrategy
    from .strategies.volatility import VolatilityStrategy

    strategy_features = {}
    strategy_labels = {}

    momentum_strategy = MomentumStrategy()
    strategy_features['momentum'] = momentum_strategy.calculate_features(df).loc[idx]
    strategy_labels['momentum'] = y

    mean_rev_strategy = MeanReversionStrategy()
    strategy_features['mean_reversion'] = mean_rev_strategy.calculate_features(df).loc[idx]
    strategy_labels['mean_reversion'] = y

    vol_strategy = VolatilityStrategy()
    strategy_features['volatility'] = vol_strategy.calculate_features(df).loc[idx]
    strategy_labels['volatility'] = y

    strategy_models_dict = train_strategy_models(
        strategy_features,
        strategy_labels,
        model_type='rf'
    )

    results['strategy_models'] = {name: model for name, (model, _) in strategy_models_dict.items()}
    results['strategy_metrics'] = {name: metrics for name, (_, metrics) in strategy_models_dict.items()}

    # Step 8: Save models
    if save_models:
        print("\n[8/8] Saving models...")
        persistence = ModelPersistence(models_dir)

        persistence.save_model(
            primary_model,
            'primary_model',
            metadata={
                'symbols': symbols,
                'start_date': start_date,
                'end_date': end_date,
                'metrics': primary_metrics
            }
        )

        persistence.save_model(meta_model, 'meta_model', metadata=meta_metrics)

        for name, model in results['strategy_models'].items():
            persistence.save_model(model, f'strategy_{name}')

        print(f"  Models saved to {models_dir}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    return results


def predict_ensemble(
    data: pd.DataFrame,
    models: Dict[str, Any],
    regime_detector: Optional[Any] = None
) -> pd.DataFrame:
    """
    Generate ensemble predictions.

    Args:
        data: Market data
        models: Dictionary of trained models
        regime_detector: Optional regime detector

    Returns:
        DataFrame with predictions
    """
    # Build features
    features = build_feature_matrix(data)

    # Get regime if detector provided
    if regime_detector is not None:
        regimes = detect_all_regimes(data)
        current_regime = regimes.iloc[-1].to_dict()
    else:
        current_regime = None

    # Get predictions from each strategy model
    strategy_models = models.get('strategy_models', {})
    predictions = run_all_strategies(
        data,
        strategy_models,
        features,
        regime=str(current_regime) if current_regime else None
    )

    # Aggregate predictions
    ensemble_pred = aggregate_strategy_predictions(predictions, method='weighted_average')

    # Create results DataFrame
    results = pd.DataFrame({
        'timestamp': [data.index[-1]],
        'signal': [ensemble_pred.signal],
        'confidence': [ensemble_pred.confidence],
        'num_strategies': [len(predictions)]
    })

    return results


def backtest_comprehensive(
    symbols: List[str],
    start_date: str,
    end_date: str,
    models: Optional[Dict] = None,
    method: str = 'simple',
    initial_capital: float = 100000
) -> Dict:
    """
    Comprehensive backtesting.

    Args:
        symbols: Symbols to backtest
        start_date: Start date
        end_date: End date
        models: Trained models (if None, will train)
        method: Backtest method ('simple', 'walk_forward')
        initial_capital: Starting capital

    Returns:
        Backtest results
    """
    print("=" * 60)
    print("PRADO9 COMPREHENSIVE BACKTEST")
    print("=" * 60)

    # Fetch data
    print("\nFetching data...")
    symbol = symbols[0]
    df = prepare_training_data(symbol, start_date, end_date)
    print(f"Data: {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Train models if not provided
    if models is None:
        print("\nNo models provided - training new models...")
        train_end = df.index[int(len(df) * 0.7)]
        models = train_ensemble(
            symbols,
            start_date,
            str(train_end.date()),
            save_models=False
        )

    # Generate features and labels
    print("\nPreparing features and labels...")
    features = build_feature_matrix(df)

    events = df.index
    labels = triple_barrier_labels(df['Close'], events, pt_sl=[1.0, 1.0])

    idx = features.index.intersection(labels.index)
    X = features.loc[idx]
    y = labels.loc[idx, 'label']

    # Run backtest
    print(f"\nRunning {method} backtest...")

    if method == 'simple':
        # Generate predictions
        predictions = pd.Series(models['primary_model'].predict(X), index=X.index)

        results = simple_backtest(
            predictions,
            df.loc[idx],
            initial_capital
        )

    elif method == 'walk_forward':
        results = walk_forward_backtest(
            models['primary_model'],
            X, y, df.loc[idx],
            train_period=252,
            test_period=63,
            initial_capital=initial_capital
        )

    else:
        raise ValueError(f"Unknown backtest method: {method}")

    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"\nInitial Capital: ${initial_capital:,.2f}")
    print(f"Final Value:     ${results['final_value']:,.2f}")
    print(f"Total Return:    {results['total_return']:.2%}")

    metrics = results['metrics']
    print(f"\nSharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio:   {metrics['sortino_ratio']:.2f}")
    print(f"Calmar Ratio:    {metrics['calmar_ratio']:.2f}")
    print(f"Max Drawdown:    {metrics['max_drawdown']:.2%}")
    print(f"Win Rate:        {metrics['win_rate']:.2%}")

    print("\n" + "=" * 60)

    return results


if __name__ == "__main__":
    # Example usage
    symbols = ["SPY"]
    start_date = "2020-01-01"
    end_date = "2023-12-31"

    # Train ensemble
    models = train_ensemble(symbols, start_date, end_date)

    # Run backtest
    results = backtest_comprehensive(symbols, start_date, end_date, models)
