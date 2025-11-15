"""
Mean reversion strategy.
Trades based on price deviations from mean.
"""
import numpy as np
import pandas as pd
from typing import Optional


class MeanReversionStrategy:
    """Mean reversion trading strategy."""

    def __init__(
        self,
        window: int = 20,
        zscore_threshold: float = 2.0,
        use_bb: bool = True
    ):
        """
        Initialize mean reversion strategy.

        Args:
            window: Window for mean calculation
            zscore_threshold: Z-score threshold for entry
            use_bb: Use Bollinger Bands instead of z-score
        """
        self.window = window
        self.zscore_threshold = zscore_threshold
        self.use_bb = use_bb

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate mean reversion signals.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series with signals (-1, 0, 1)
        """
        if self.use_bb:
            return self._bollinger_signals(df)
        else:
            return self._zscore_signals(df)

    def _zscore_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate signals using z-score."""
        # Calculate z-score
        ma = df['Close'].rolling(self.window).mean()
        std = df['Close'].rolling(self.window).std()
        zscore = (df['Close'] - ma) / (std + 1e-10)

        # Generate signals
        signals = pd.Series(0, index=df.index)
        signals[zscore < -self.zscore_threshold] = 1  # Oversold - buy
        signals[zscore > self.zscore_threshold] = -1  # Overbought - sell

        return signals

    def _bollinger_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate signals using Bollinger Bands."""
        # Calculate Bollinger Bands
        ma = df['Close'].rolling(self.window).mean()
        std = df['Close'].rolling(self.window).std()
        upper = ma + 2 * std
        lower = ma - 2 * std

        # Generate signals
        signals = pd.Series(0, index=df.index)
        signals[df['Close'] < lower] = 1  # Below lower band - buy
        signals[df['Close'] > upper] = -1  # Above upper band - sell

        return signals

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate mean reversion features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with features
        """
        features = pd.DataFrame(index=df.index)

        # Z-scores at multiple windows
        for window in [10, 20, 50]:
            ma = df['Close'].rolling(window).mean()
            std = df['Close'].rolling(window).std()
            features[f'zscore_{window}'] = (df['Close'] - ma) / (std + 1e-10)

        # Bollinger Band position
        ma = df['Close'].rolling(self.window).mean()
        std = df['Close'].rolling(self.window).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        features['bb_position'] = (df['Close'] - lower) / (upper - lower + 1e-10)

        # Distance from mean
        for window in [10, 20, 50]:
            ma = df['Close'].rolling(window).mean()
            features[f'mean_distance_{window}'] = (df['Close'] - ma) / ma

        # RSI (mean reversion indicator)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        features['rsi'] = 100 - (100 / (1 + rs))

        # Stochastic
        low_min = df['Low'].rolling(14).min()
        high_max = df['High'].rolling(14).max()
        features['stoch'] = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-10)

        # Mean reversion speed (autocorrelation)
        returns = df['Close'].pct_change()
        features['mean_reversion_speed'] = returns.rolling(20).apply(
            lambda x: x.autocorr(1) if len(x) > 1 else 0
        )

        return features

    def get_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate signal strength (0 to 1).

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series with signal strengths
        """
        # Calculate z-score
        ma = df['Close'].rolling(self.window).mean()
        std = df['Close'].rolling(self.window).std()
        zscore = abs((df['Close'] - ma) / (std + 1e-10))

        # Normalize to [0, 1]
        strength = zscore / (zscore.rolling(50).max() + 1e-10)

        return strength.clip(0, 1)

    def is_favorable_regime(self, regime: str) -> bool:
        """
        Check if current regime favors mean reversion.

        Args:
            regime: Current regime label

        Returns:
            True if regime is favorable
        """
        favorable_regimes = ['ranging', 'ranging_low_vol', 'ranging_normal_vol']
        return any(fav in regime for fav in favorable_regimes)
