"""
Thompson Sampling Bandit for strategy selection.
Learns which strategies perform best in different regimes.

Implements production-grade multi-armed bandit similar to what
Renaissance Technologies / Medallion Fund would use.
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime


@dataclass
class ArmStats:
    """Statistics for a single arm (strategy, regime) pair."""
    alpha: float = 1.0  # Successes (Beta prior parameter)
    beta: float = 1.0   # Failures (Beta prior parameter)
    pulls: int = 0      # Number of times selected
    total_reward: float = 0.0  # Cumulative reward
    last_updated: Optional[str] = None  # Timestamp

    @property
    def win_rate(self) -> float:
        """Empirical win rate."""
        if self.alpha + self.beta <= 2:
            return 0.5  # Prior mean
        return self.alpha / (self.alpha + self.beta)

    @property
    def confidence_interval(self) -> Tuple[float, float]:
        """95% credible interval."""
        # Beta distribution percentiles
        lower = np.percentile(np.random.beta(self.alpha, self.beta, 10000), 2.5)
        upper = np.percentile(np.random.beta(self.alpha, self.beta, 10000), 97.5)
        return (lower, upper)


@dataclass
class ThompsonSamplingBandit:
    """
    Thompson Sampling bandit for multi-armed strategy selection.

    Key features (Medallion-style):
    1. Bayesian updating with Beta-Binomial conjugate prior
    2. Per-regime learning (contextual bandit)
    3. Exploration-exploitation balance via posterior sampling
    4. Decay for non-stationary environments
    5. Persistent state tracking
    """

    arms: Dict[Tuple[str, str], ArmStats] = field(default_factory=dict)
    decay_factor: float = 0.995  # Exponential decay for non-stationarity
    min_alpha: float = 1.0  # Floor to maintain exploration
    min_beta: float = 1.0

    def select_strategies(
        self,
        predictions: List,
        regime: str,
        n_select: int = 3,
        exploration_rate: float = 0.1
    ) -> List:
        """
        Select top N strategies using Thompson Sampling.

        Medallion approach:
        1. Sample from posterior Beta distribution for each arm
        2. Select top N based on samples (probabilistic selection)
        3. Occasionally force exploration of undersampled arms

        Args:
            predictions: List of Prediction objects
            regime: Current regime
            n_select: Number of strategies to select
            exploration_rate: Probability of forced exploration

        Returns:
            List of selected Prediction objects (reordered by Thompson sample)
        """
        if not predictions:
            return []

        # Force exploration with small probability
        if np.random.random() < exploration_rate:
            # Random selection for exploration
            np.random.shuffle(predictions)
            return predictions[:n_select]

        # Thompson Sampling: Sample from posterior Beta distributions
        samples = []
        for pred in predictions:
            key = (pred.strategy_name, regime)

            # Initialize arm if not seen
            if key not in self.arms:
                self.arms[key] = ArmStats()

            arm = self.arms[key]

            # Sample from Beta(alpha, beta)
            thompson_sample = np.random.beta(arm.alpha, arm.beta)

            samples.append({
                'prediction': pred,
                'sample': thompson_sample,
                'win_rate': arm.win_rate,
                'pulls': arm.pulls
            })

        # Sort by Thompson samples (highest first)
        samples.sort(key=lambda x: x['sample'], reverse=True)

        # Select top N
        selected = [s['prediction'] for s in samples[:n_select]]

        return selected

    def update(
        self,
        strategy_name: str,
        regime: str,
        reward: float,
        apply_decay: bool = True
    ):
        """
        Update bandit with observed reward.

        Medallion approach:
        1. Bayesian update (conjugate Beta-Binomial)
        2. Apply exponential decay to all arms (handle non-stationarity)
        3. Maintain exploration floor

        Args:
            strategy_name: Name of strategy
            regime: Market regime
            reward: Reward (0 to 1, can be continuous)
            apply_decay: Whether to decay other arms
        """
        key = (strategy_name, regime)

        # Initialize if new arm
        if key not in self.arms:
            self.arms[key] = ArmStats()

        arm = self.arms[key]

        # Bayesian update (treat reward as Bernoulli trial)
        # If reward is continuous [0,1], treat as fractional success
        arm.alpha += reward
        arm.beta += (1.0 - reward)
        arm.pulls += 1
        arm.total_reward += reward
        arm.last_updated = datetime.now().isoformat()

        # Apply decay to ALL arms in this regime (non-stationarity)
        if apply_decay:
            for (strat, reg), arm_stats in self.arms.items():
                if reg == regime:
                    # Decay towards prior (geometric decay)
                    arm_stats.alpha = max(
                        self.min_alpha,
                        self.min_alpha + (arm_stats.alpha - self.min_alpha) * self.decay_factor
                    )
                    arm_stats.beta = max(
                        self.min_beta,
                        self.min_beta + (arm_stats.beta - self.min_beta) * self.decay_factor
                    )

    def batch_update(
        self,
        outcomes: List[Dict[str, any]]
    ):
        """
        Batch update for backtest replay.

        Args:
            outcomes: List of dicts with keys:
                - strategy_name: str
                - regime: str
                - reward: float
                - timestamp: str (optional)
        """
        for outcome in outcomes:
            self.update(
                outcome['strategy_name'],
                outcome['regime'],
                outcome['reward'],
                apply_decay=False  # Apply decay once at end
            )

        # Single decay pass after batch
        for arm in self.arms.values():
            arm.alpha = max(
                self.min_alpha,
                self.min_alpha + (arm.alpha - self.min_alpha) * self.decay_factor
            )
            arm.beta = max(
                self.min_beta,
                self.min_beta + (arm.beta - self.min_beta) * self.decay_factor
            )

    def get_arm_stats(self, strategy_name: str, regime: str) -> ArmStats:
        """Get statistics for a specific arm."""
        key = (strategy_name, regime)
        if key not in self.arms:
            self.arms[key] = ArmStats()
        return self.arms[key]

    def get_regime_rankings(self, regime: str) -> List[Dict]:
        """
        Get strategy rankings for a regime.

        Returns:
            List of dicts sorted by win_rate, containing:
                - strategy_name
                - win_rate
                - pulls
                - confidence_interval
        """
        rankings = []
        for (strategy, reg), arm in self.arms.items():
            if reg == regime:
                ci = arm.confidence_interval
                rankings.append({
                    'strategy_name': strategy,
                    'win_rate': arm.win_rate,
                    'pulls': arm.pulls,
                    'alpha': arm.alpha,
                    'beta': arm.beta,
                    'ci_lower': ci[0],
                    'ci_upper': ci[1],
                    'total_reward': arm.total_reward
                })

        rankings.sort(key=lambda x: x['win_rate'], reverse=True)
        return rankings

    def save(self, symbol: str):
        """Save bandit state to disk."""
        from ..models.persistence import get_models_dir

        models_dir = get_models_dir()
        symbol_dir = models_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)

        bandit_path = symbol_dir / 'thompson_bandit.pkl'

        # Convert to serializable format
        state = {
            'arms': {
                f"{strat}_{reg}": asdict(arm)
                for (strat, reg), arm in self.arms.items()
            },
            'decay_factor': self.decay_factor,
            'min_alpha': self.min_alpha,
            'min_beta': self.min_beta
        }

        with open(bandit_path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, symbol: str) -> 'ThompsonSamplingBandit':
        """Load bandit state from disk."""
        from ..models.persistence import get_models_dir

        try:
            models_dir = get_models_dir()
            bandit_path = models_dir / symbol / 'thompson_bandit.pkl'

            if not bandit_path.exists():
                # Return fresh bandit
                return cls()

            with open(bandit_path, 'rb') as f:
                state = pickle.load(f)

            # Reconstruct arms
            arms = {}
            for key, arm_dict in state['arms'].items():
                strategy, regime = key.rsplit('_', 1)
                arms[(strategy, regime)] = ArmStats(**arm_dict)

            return cls(
                arms=arms,
                decay_factor=state.get('decay_factor', 0.995),
                min_alpha=state.get('min_alpha', 1.0),
                min_beta=state.get('min_beta', 1.0)
            )

        except Exception as e:
            print(f"Warning: Could not load bandit for {symbol}: {e}")
            return cls()

    def reset_arm(self, strategy_name: str, regime: str):
        """Reset a specific arm to prior."""
        key = (strategy_name, regime)
        self.arms[key] = ArmStats()

    def get_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all arms."""
        rows = []
        for (strategy, regime), arm in self.arms.items():
            ci = arm.confidence_interval
            rows.append({
                'strategy': strategy,
                'regime': regime,
                'win_rate': arm.win_rate,
                'pulls': arm.pulls,
                'alpha': arm.alpha,
                'beta': arm.beta,
                'ci_lower': ci[0],
                'ci_upper': ci[1],
                'total_reward': arm.total_reward,
                'last_updated': arm.last_updated
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(['regime', 'win_rate'], ascending=[True, False])

        return df
