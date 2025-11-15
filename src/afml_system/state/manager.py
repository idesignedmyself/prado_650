"""
Bandit state management for PRADO9 system.
Implements Thompson Sampling for strategy selection.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class BanditArm:
    """Represents a single bandit arm (strategy)."""
    name: str
    alpha: float = 1.0  # Success count (Beta distribution)
    beta: float = 1.0   # Failure count (Beta distribution)
    total_trials: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0

    def sample(self) -> float:
        """Sample from Beta distribution."""
        return np.random.beta(self.alpha, self.beta)

    def update(self, reward: float):
        """
        Update arm statistics.

        Args:
            reward: Reward value (-1 to 1, where 1 is best)
        """
        # Convert reward to [0, 1] range
        normalized_reward = (reward + 1) / 2

        # Update Beta distribution parameters
        if normalized_reward > 0.5:
            self.alpha += normalized_reward
        else:
            self.beta += (1 - normalized_reward)

        # Update statistics
        self.total_trials += 1
        self.total_reward += reward
        self.avg_reward = self.total_reward / self.total_trials


class BanditStateManager:
    """Manages multi-armed bandit state for strategy selection."""

    def __init__(self, state_file: str = "bandit_state.json"):
        """
        Initialize bandit state manager.

        Args:
            state_file: Path to state persistence file
        """
        self.state_file = state_file
        self.arms: Dict[str, BanditArm] = {}
        self.history: List[Dict] = []
        self._load_state()

    def _load_state(self):
        """Load state from file if exists."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)

                # Load arms
                for arm_data in data.get('arms', []):
                    arm = BanditArm(**arm_data)
                    self.arms[arm.name] = arm

                # Load history
                self.history = data.get('history', [])

            except Exception as e:
                print(f"Error loading bandit state: {e}")
                self.arms = {}
                self.history = []

    def save_state(self):
        """Save state to file."""
        try:
            Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)

            data = {
                'arms': [asdict(arm) for arm in self.arms.values()],
                'history': self.history
            }

            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"Error saving bandit state: {e}")

    def register_arm(self, name: str, initial_alpha: float = 1.0, initial_beta: float = 1.0):
        """
        Register a new bandit arm.

        Args:
            name: Arm/strategy name
            initial_alpha: Initial alpha parameter
            initial_beta: Initial beta parameter
        """
        if name not in self.arms:
            self.arms[name] = BanditArm(
                name=name,
                alpha=initial_alpha,
                beta=initial_beta
            )

    def select_arm(self, method: str = "thompson") -> str:
        """
        Select an arm using Thompson Sampling.

        Args:
            method: Selection method ("thompson", "ucb", "epsilon_greedy")

        Returns:
            Selected arm name
        """
        if not self.arms:
            raise ValueError("No arms registered")

        if method == "thompson":
            return self._thompson_sampling()
        elif method == "ucb":
            return self._ucb_selection()
        elif method == "epsilon_greedy":
            return self._epsilon_greedy(epsilon=0.1)
        else:
            raise ValueError(f"Unknown selection method: {method}")

    def _thompson_sampling(self) -> str:
        """Thompson Sampling selection."""
        samples = {name: arm.sample() for name, arm in self.arms.items()}
        return max(samples, key=samples.get)

    def _ucb_selection(self, c: float = 2.0) -> str:
        """Upper Confidence Bound selection."""
        total_trials = sum(arm.total_trials for arm in self.arms.values())
        if total_trials == 0:
            return list(self.arms.keys())[0]

        ucb_values = {}
        for name, arm in self.arms.items():
            if arm.total_trials == 0:
                ucb_values[name] = float('inf')
            else:
                exploration = c * np.sqrt(np.log(total_trials) / arm.total_trials)
                ucb_values[name] = arm.avg_reward + exploration

        return max(ucb_values, key=ucb_values.get)

    def _epsilon_greedy(self, epsilon: float = 0.1) -> str:
        """Epsilon-greedy selection."""
        if np.random.random() < epsilon:
            # Explore: random selection
            return np.random.choice(list(self.arms.keys()))
        else:
            # Exploit: select best arm
            return max(self.arms.items(), key=lambda x: x[1].avg_reward)[0]

    def update_arm(self, name: str, reward: float):
        """
        Update arm with reward.

        Args:
            name: Arm name
            reward: Reward value (-1 to 1)
        """
        if name not in self.arms:
            raise ValueError(f"Arm not registered: {name}")

        self.arms[name].update(reward)

        # Add to history
        self.history.append({
            'arm': name,
            'reward': reward,
            'alpha': self.arms[name].alpha,
            'beta': self.arms[name].beta,
            'avg_reward': self.arms[name].avg_reward
        })

        # Save state after update
        self.save_state()

    def get_arm_stats(self, name: str) -> Optional[BanditArm]:
        """Get statistics for an arm."""
        return self.arms.get(name)

    def get_all_stats(self) -> Dict[str, Dict]:
        """Get statistics for all arms."""
        return {
            name: {
                'alpha': arm.alpha,
                'beta': arm.beta,
                'total_trials': arm.total_trials,
                'avg_reward': arm.avg_reward,
                'expected_value': arm.alpha / (arm.alpha + arm.beta)
            }
            for name, arm in self.arms.items()
        }

    def reset_arm(self, name: str):
        """Reset an arm to initial state."""
        if name in self.arms:
            self.arms[name] = BanditArm(name=name)
            self.save_state()

    def reset_all(self):
        """Reset all arms."""
        self.arms = {}
        self.history = []
        self.save_state()

    def get_best_arm(self) -> str:
        """Get arm with highest average reward."""
        if not self.arms:
            raise ValueError("No arms registered")
        return max(self.arms.items(), key=lambda x: x[1].avg_reward)[0]

    def get_arm_probabilities(self) -> Dict[str, float]:
        """Get probability of each arm being best (via sampling)."""
        n_samples = 10000
        wins = {name: 0 for name in self.arms.keys()}

        for _ in range(n_samples):
            samples = {name: arm.sample() for name, arm in self.arms.items()}
            winner = max(samples, key=samples.get)
            wins[winner] += 1

        return {name: count / n_samples for name, count in wins.items()}
