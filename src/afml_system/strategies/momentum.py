"""
Momentum strategy.
Trend-following strategy based on price momentum.
"""
import numpy as np
import pandas as pd
from typing import Optional


class MomentumStrategy:
    """Momentum-based trading strategy."""

    def __init__(
        self,
        lookback: int = 20,
        threshold: float = 0.02,
        use_volume_confirmation: bool = True
    ):
        """
        Initialize momentum strategy.

        Args:
            lookback: Lookback period for momentum
            threshold: Minimum momentum threshold
            use_volume_confirmation: Require volume confirmation
        """
        self.lookback = lookback
        self.threshold = threshold
        self.use_volume_confirmation = use_volume_confirmation

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate momentum signals.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series with signals (-1, 0, 1)
        """
        # Calculate momentum
        momentum = df['Close'].pct_change(self.lookback)

        # Initialize signals
        signals = pd.Series(0, index=df.index)

        # Generate signals based on momentum
        signals[momentum > self.threshold] = 1
        signals[momentum < -self.threshold] = -1

        # Volume confirmation
        if self.use_volume_confirmation:
            vol_ma = df['Volume'].rolling(self.lookback).mean()
            high_volume = df['Volume'] > vol_ma

            # Only keep signals with volume confirmation
            signals = signals * high_volume.astype(int)

        return signals

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum-related features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with features
        """
        features = pd.DataFrame(index=df.index)

        # Momentum at multiple timeframes
        for period in [5, 10, 20, 50]:
            features[f'momentum_{period}'] = df['Close'].pct_change(period)

        # Rate of change
        features['roc_20'] = (df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20)

        # Moving average crossovers
        features['sma_20'] = df['Close'].rolling(20).mean()
        features['sma_50'] = df['Close'].rolling(50).mean()
        features['ma_cross'] = (features['sma_20'] > features['sma_50']).astype(int)

        # Price vs moving averages
        features['price_sma_ratio'] = df['Close'] / features['sma_20']

        # Momentum strength
        features['momentum_strength'] = abs(features['momentum_20'])

        # Trend consistency (percentage of positive periods)
        returns = df['Close'].pct_change()
        features['trend_consistency'] = returns.rolling(20).apply(
            lambda x: (x > 0).sum() / len(x)
        )

        # Volume trend
        features['volume_trend'] = df['Volume'].pct_change(self.lookback)

        return features

    def get_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate signal strength (0 to 1).

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series with signal strengths
        """
        momentum = df['Close'].pct_change(self.lookback)

        # Normalize momentum to [0, 1]
        strength = abs(momentum) / (abs(momentum).rolling(50).max() + 1e-10)

        return strength.clip(0, 1)

    def is_favorable_regime(self, regime: str) -> bool:
        """
        Check if current regime favors momentum strategy.

        Args:
            regime: Current regime label

        Returns:
            True if regime is favorable
        """
        favorable_regimes = ['trending', 'trending_normal_vol', 'trending_high_volume']
        return any(fav in regime for fav in favorable_regimes)
