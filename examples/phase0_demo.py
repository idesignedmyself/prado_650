"""
Phase 0 Demo: Configuration and State Management

This demo shows how to:
1. Initialize configuration manager
2. Load and update configuration
3. Initialize bandit state manager for strategy selection
4. Demonstrate state persistence
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from afml_system.config import ConfigManager, PRADO9Config
from afml_system.state import BanditStateManager


def demo_config_management():
    """Demonstrate configuration management."""
    print("=" * 60)
    print("Phase 0: Configuration and State Management")
    print("=" * 60)

    # Initialize config manager with default configuration
    print("\n1. Initializing ConfigManager with defaults...")
    config_mgr = ConfigManager()
    config = config_mgr.config

    print(f"   - Data symbols: {config.data.symbols}")
    print(f"   - Data start date: {config.data.start_date}")
    print(f"   - Data end date: {config.data.end_date}")
    print(f"   - Bar type: {config.data.bar_type}")
    print(f"   - Bar threshold: ${config.data.bar_threshold:,.0f}")

    # Show labeling config
    print(f"\n2. Labeling Configuration:")
    print(f"   - Profit taking/stop loss multipliers: {config.labeling.pt_sl}")
    print(f"   - Minimum return threshold: {config.labeling.min_ret}")
    print(f"   - Number of threads: {config.labeling.num_threads}")
    print(f"   - Vertical barrier (days): {config.labeling.vertical_barrier_days}")

    # Show feature config
    print(f"\n3. Feature Configuration:")
    print(f"   - Lookback periods: {config.features.lookback_periods}")
    print(f"   - Volatility window: {config.features.volatility_window}")
    print(f"   - Use microstructure features: {config.features.use_microstructure}")
    print(f"   - Use technical features: {config.features.use_technical}")

    # Show regime config
    print(f"\n4. Regime Detection Configuration:")
    print(f"   - ADX period: {config.regime.adx_period}")
    print(f"   - ADX threshold: {config.regime.adx_threshold}")
    print(f"   - EMA period: {config.regime.ema_period}")
    print(f"   - Volume window: {config.regime.volume_window}")

    # Show model config
    print(f"\n5. Model Configuration:")
    print(f"   - Number of estimators: {config.model.n_estimators}")
    print(f"   - Max tree depth: {config.model.max_depth}")
    print(f"   - Learning rate: {config.model.learning_rate}")
    print(f"   - CV folds: {config.model.cv_folds}")
    print(f"   - Test size: {config.model.test_size}")

    # Show strategy config
    print(f"\n6. Strategy Configuration:")
    print(f"   - Momentum lookback: {config.strategy.momentum_lookback}")
    print(f"   - Mean reversion zscore: {config.strategy.mean_reversion_zscore}")
    print(f"   - Volatility threshold: {config.strategy.volatility_threshold}")
    print(f"   - Pairs window: {config.strategy.pairs_window}")
    print(f"   - Scalping threshold: {config.strategy.scalping_threshold}")

    # Show allocation config
    print(f"\n7. Allocation Configuration:")
    print(f"   - Max position size: {config.allocation.max_position_size:.1%}")
    print(f"   - Kelly fraction: {config.allocation.kelly_fraction}")
    print(f"   - Volatility target: {config.allocation.volatility_target:.1%}")
    print(f"   - Leverage limit: {config.allocation.leverage_limit}x")

    # Show risk config
    print(f"\n8. Risk Configuration:")
    print(f"   - Max drawdown: {config.risk.max_drawdown:.1%}")
    print(f"   - Max position size: {config.risk.max_position_size:.1%}")
    print(f"   - Stop loss: {config.risk.stop_loss:.1%}")
    print(f"   - Take profit: {config.risk.take_profit:.1%}")

    # Update configuration
    print(f"\n9. Updating Configuration:")
    config_mgr.update("data.symbols", ["SPY", "QQQ", "IWM"])
    config_mgr.update("data.start_date", "2022-01-01")
    print(f"   - New symbols: {config_mgr.get('data.symbols')}")
    print(f"   - New start date: {config_mgr.get('data.start_date')}")

    return config_mgr


def demo_bandit_state_management():
    """Demonstrate bandit state management."""
    print("\n" + "=" * 60)
    print("Phase 0: Bandit State Management")
    print("=" * 60)

    # Initialize bandit state manager
    print("\n1. Initializing BanditStateManager...")
    state_mgr = BanditStateManager(state_file="/tmp/prado9_bandit_demo.json")

    # Register strategies as arms
    print("\n2. Registering strategy arms:")
    strategies = ["momentum", "mean_reversion", "volatility", "pairs_trading"]
    for strategy in strategies:
        state_mgr.register_arm(strategy)
        print(f"   - Registered: {strategy}")

    # Simulate strategy performance
    print("\n3. Simulating strategy performance over 10 rounds:")
    import numpy as np
    np.random.seed(42)

    for round_num in range(10):
        # Select strategy using Thompson Sampling
        selected = state_mgr.select_arm(method="thompson")

        # Simulate returns
        if selected == "momentum":
            reward = np.random.normal(0.002, 0.015)  # Small positive drift
        elif selected == "mean_reversion":
            reward = np.random.normal(0.001, 0.012)  # Lower drift
        elif selected == "volatility":
            reward = np.random.normal(0.000, 0.020)  # High volatility
        else:  # pairs_trading
            reward = np.random.normal(0.003, 0.010)  # Good risk-reward

        # Update arm with reward
        state_mgr.update_arm(selected, reward)
        print(f"   Round {round_num + 1}: {selected:20s} -> Reward: {reward:7.4f}")

    # Get all statistics
    print("\n4. Final Arm Statistics:")
    stats = state_mgr.get_all_stats()
    for name, stat in stats.items():
        print(f"   {name}:")
        print(f"      - Total trials: {stat['total_trials']}")
        print(f"      - Avg reward: {stat['avg_reward']:7.4f}")
        print(f"      - Expected value: {stat['expected_value']:.4f}")
        print(f"      - Alpha (successes): {stat['alpha']:.2f}")
        print(f"      - Beta (failures): {stat['beta']:.2f}")

    # Get arm probabilities
    print("\n5. Probability of Each Arm Being Best (10k samples):")
    probs = state_mgr.get_arm_probabilities()
    for name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {name}: {prob:.2%}")

    # Get best arm
    print(f"\n6. Best Performing Arm: {state_mgr.get_best_arm()}")

    return state_mgr


if __name__ == "__main__":
    # Run config demo
    config_mgr = demo_config_management()

    # Run bandit state demo
    state_mgr = demo_bandit_state_management()

    print("\n" + "=" * 60)
    print("Phase 0 Demo Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("- Run phase1_demo.py to learn about CUSUM and bar generation")
    print("- Run phase2_demo.py to learn about triple barrier labeling")
    print("- Run phase3_demo.py to learn about feature building")
    print("- Run phase4_demo.py to learn about regime detection")
    print("- Run phase5_demo.py to learn about strategy predictions")
    print("- Run phase6_demo.py to learn about model training")
    print("- Run full_pipeline_demo.py to see the complete pipeline")
