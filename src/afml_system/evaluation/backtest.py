"""
Backtesting methods.
Implements 4 backtest approaches: simple, walk-forward, monte carlo, regime-based.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from ..execution.engine import ExecutionEngine
from ..execution.risk import RiskManager
from .metrics import get_all_metrics


def simple_backtest(
    signals: pd.Series,
    prices: pd.DataFrame,
    initial_capital: float = 100000,
    commission_rate: float = 0.001
) -> Dict:
    """
    Simple backtest with single asset.

    Args:
        signals: Trading signals (-1, 0, 1)
        prices: Price DataFrame with Close column
        initial_capital: Starting capital
        commission_rate: Commission rate

    Returns:
        Backtest results dictionary
    """
    # Initialize
    portfolio_value = initial_capital
    positions = []
    returns = []

    execution_engine = ExecutionEngine(commission_rate=commission_rate)

    # Align signals and prices
    idx = signals.index.intersection(prices.index)
    signals = signals.loc[idx]
    prices = prices.loc[idx]

    current_position = 0

    for timestamp in idx:
        signal = signals.loc[timestamp]
        price = prices.loc[timestamp, 'Close']

        # Calculate target position
        target_position = signal * 0.1  # 10% of portfolio

        # Execute if position change needed
        if abs(target_position - current_position) > 0.01:
            trade = execution_engine.execute_trade(
                timestamp, 'ASSET', signal, price,
                portfolio_value, abs(target_position - current_position)
            )

            if trade:
                # Update position
                if signal > 0:
                    current_position += abs(target_position - current_position)
                elif signal < 0:
                    current_position -= abs(target_position - current_position)
                else:
                    current_position = 0

        # Calculate portfolio value
        position_value = current_position * portfolio_value * price / prices.loc[idx[0], 'Close']
        cash = portfolio_value * (1 - abs(current_position))
        portfolio_value = cash + position_value

        positions.append(current_position)

        # Calculate return
        if len(positions) > 1:
            ret = (portfolio_value - initial_capital) / initial_capital
            returns.append(ret)

    # Calculate metrics
    returns_series = pd.Series(returns, index=idx[1:]) if returns else pd.Series()
    metrics = get_all_metrics(returns_series.pct_change().fillna(0))

    results = {
        'final_value': portfolio_value,
        'total_return': (portfolio_value - initial_capital) / initial_capital,
        'metrics': metrics,
        'trades': execution_engine.get_trade_history(),
        'execution_stats': execution_engine.get_execution_stats()
    }

    return results


def walk_forward_backtest(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    prices: pd.DataFrame,
    train_period: int = 252,
    test_period: int = 63,
    initial_capital: float = 100000
) -> Dict:
    """
    Walk-forward backtest with retraining.

    Args:
        model: ML model to train
        X: Features
        y: Labels
        prices: Price data
        train_period: Training window size
        test_period: Test window size
        initial_capital: Starting capital

    Returns:
        Backtest results
    """
    results = {
        'predictions': [],
        'actual': [],
        'returns': [],
        'dates': []
    }

    # Walk forward
    start = train_period
    while start + test_period < len(X):
        # Split data
        train_start = start - train_period
        train_end = start
        test_end = start + test_period

        X_train = X.iloc[train_start:train_end]
        y_train = y.iloc[train_start:train_end]
        X_test = X.iloc[start:test_end]
        y_test = y.iloc[start:test_end]

        # Train model
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Store results
        results['predictions'].extend(y_pred)
        results['actual'].extend(y_test.values)
        results['dates'].extend(X_test.index)

        # Move forward
        start += test_period

    # Convert to series
    predictions = pd.Series(results['predictions'], index=results['dates'])
    actual = pd.Series(results['actual'], index=results['dates'])

    # Run backtest on predictions
    backtest_results = simple_backtest(
        predictions,
        prices.loc[predictions.index],
        initial_capital
    )

    backtest_results['walk_forward_stats'] = {
        'num_windows': len(results['dates']) // test_period,
        'train_period': train_period,
        'test_period': test_period
    }

    return backtest_results


def monte_carlo_backtest(
    signals: pd.Series,
    prices: pd.DataFrame,
    n_simulations: int = 1000,
    initial_capital: float = 100000
) -> Dict:
    """
    Monte Carlo backtest with randomized trade sequences.

    Args:
        signals: Trading signals
        prices: Price data
        n_simulations: Number of simulations
        initial_capital: Starting capital

    Returns:
        Monte Carlo results
    """
    results = []

    for i in range(n_simulations):
        # Randomize trade sequence (bootstrap)
        idx = np.random.choice(signals.index, size=len(signals), replace=True)
        randomized_signals = signals.loc[idx]
        randomized_signals.index = signals.index

        # Run backtest
        sim_result = simple_backtest(
            randomized_signals,
            prices,
            initial_capital
        )

        results.append({
            'final_value': sim_result['final_value'],
            'total_return': sim_result['total_return'],
            'sharpe': sim_result['metrics']['sharpe_ratio']
        })

    # Aggregate results
    final_values = [r['final_value'] for r in results]
    total_returns = [r['total_return'] for r in results]
    sharpes = [r['sharpe'] for r in results]

    mc_results = {
        'mean_final_value': np.mean(final_values),
        'median_final_value': np.median(final_values),
        'std_final_value': np.std(final_values),
        'mean_return': np.mean(total_returns),
        'median_return': np.median(total_returns),
        'std_return': np.std(total_returns),
        'mean_sharpe': np.mean(sharpes),
        'percentile_5': np.percentile(final_values, 5),
        'percentile_95': np.percentile(final_values, 95),
        'probability_profit': sum(1 for r in total_returns if r > 0) / len(total_returns)
    }

    return mc_results


def regime_based_backtest(
    strategy_models: Dict[str, Any],
    features: pd.DataFrame,
    regime: pd.Series,
    prices: pd.DataFrame,
    regime_strategy_map: Dict[str, List[str]],
    initial_capital: float = 100000
) -> Dict:
    """
    Backtest with regime-based strategy selection.

    Args:
        strategy_models: Dictionary of strategy models
        features: Feature matrix
        regime: Regime series
        prices: Price data
        regime_strategy_map: Mapping of regime to strategies
        initial_capital: Starting capital

    Returns:
        Backtest results
    """
    # Align data
    idx = features.index.intersection(regime.index).intersection(prices.index)
    features = features.loc[idx]
    regime = regime.loc[idx]
    prices = prices.loc[idx]

    # Generate predictions for each regime
    all_predictions = pd.Series(0, index=idx)

    for regime_label in regime.unique():
        regime_mask = regime == regime_label

        # Get suitable strategies for this regime
        if regime_label in regime_strategy_map:
            suitable_strategies = regime_strategy_map[regime_label]
        else:
            continue

        # Get predictions from each strategy
        strategy_predictions = []

        for strategy_name in suitable_strategies:
            if strategy_name in strategy_models:
                model = strategy_models[strategy_name]
                pred = model.predict(features[regime_mask])
                strategy_predictions.append(pred)

        # Average predictions
        if strategy_predictions:
            avg_pred = np.mean(strategy_predictions, axis=0)
            all_predictions.loc[regime_mask] = avg_pred

    # Run backtest
    results = simple_backtest(all_predictions, prices, initial_capital)

    results['regime_stats'] = {
        'num_regimes': regime.nunique(),
        'regime_counts': regime.value_counts().to_dict()
    }

    return results
