"""
Scalping strategy.
High-frequency short-term trading on small price movements.
"""
import numpy as np
import pandas as pd
from typing import Optional


class ScalpingStrategy:
    """Scalping trading strategy."""

    def __init__(
        self,
        tick_threshold: float = 0.005,
        holding_period: int = 1,
        use_spread: bool = True
    ):
        """
        Initialize scalping strategy.

        Args:
            tick_threshold: Minimum price movement for signal
            holding_period: Maximum holding period (bars)
            use_spread: Consider bid-ask spread
        """
        self.tick_threshold = tick_threshold
        self.holding_period = holding_period
        self.use_spread = use_spread

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate scalping signals.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series with signals (-1, 0, 1)
        """
        # Short-term price movements
        price_change = df['Close'].pct_change()

        # Tick direction
        tick_direction = pd.Series(0, index=df.index)
        tick_direction[price_change > 0] = 1
        tick_direction[price_change < 0] = -1

        # Microstructure signals
        signals = pd.Series(0, index=df.index)

        # Order flow imbalance
        ofi = self._calculate_ofi(df)

        # Generate signals on OFI and price momentum
        short_momentum = price_change.rolling(3).mean()

        signals[(ofi > 0) & (short_momentum > self.tick_threshold)] = 1
        signals[(ofi < 0) & (short_momentum < -self.tick_threshold)] = -1

        # Consider spread if enabled
        if self.use_spread:
            spread = self._estimate_spread(df)
            wide_spread = spread > spread.rolling(20).mean()
            signals[wide_spread] = 0  # No trading when spread is wide

        return signals

    def _calculate_ofi(self, df: pd.DataFrame, window: int = 5) -> pd.Series:
        """Calculate Order Flow Imbalance."""
        price_change = df['Close'].diff()

        tick_direction = pd.Series(0, index=df.index)
        tick_direction[price_change > 0] = 1
        tick_direction[price_change < 0] = -1

        signed_volume = tick_direction * df['Volume']
        ofi = signed_volume.rolling(window).sum()

        return ofi

    def _estimate_spread(self, df: pd.DataFrame) -> pd.Series:
        """Estimate bid-ask spread from high-low."""
        spread = (df['High'] - df['Low']) / df['Close']
        return spread

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate scalping features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with features
        """
        features = pd.DataFrame(index=df.index)

        # Very short-term momentum
        for period in [1, 2, 3, 5]:
            features[f'momentum_{period}'] = df['Close'].pct_change(period)

        # Tick direction
        price_change = df['Close'].diff()
        features['tick_direction'] = np.sign(price_change)

        # Order flow imbalance
        for window in [3, 5, 10]:
            features[f'ofi_{window}'] = self._calculate_ofi(df, window)

        # Spread estimate
        features['spread'] = self._estimate_spread(df)
        features['spread_ma'] = features['spread'].rolling(20).mean()
        features['spread_ratio'] = features['spread'] / features['spread_ma']

        # Volume features
        features['volume_surge'] = df['Volume'] / df['Volume'].rolling(10).mean()

        # Price velocity (rate of change)
        features['price_velocity'] = df['Close'].diff() / df['Close'].shift()

        # Microstructure noise
        features['microstructure_noise'] = self._calculate_noise(df)

        # Tick imbalance
        for window in [5, 10, 20]:
            features[f'tick_imbalance_{window}'] = features['tick_direction'].rolling(window).sum()

        # Recent high/low
        features['near_high'] = (df['Close'] / df['High'].rolling(5).max()).fillna(1)
        features['near_low'] = (df['Close'] / df['Low'].rolling(5).min()).fillna(1)

        return features

    def _calculate_noise(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate microstructure noise."""
        # Variance ratio
        returns = df['Close'].pct_change()

        var_1 = returns.rolling(window).var()
        var_2 = returns.rolling(window * 2).var()

        noise = 1 - (var_2 / (2 * var_1 + 1e-10))

        return noise

    def get_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate signal strength (0 to 1).

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series with signal strengths
        """
        # Based on OFI magnitude
        ofi = self._calculate_ofi(df)
        ofi_abs = abs(ofi)

        # Normalize
        ofi_max = ofi_abs.rolling(50).max()
        strength = ofi_abs / (ofi_max + 1e-10)

        return strength.clip(0, 1)

    def is_favorable_regime(self, regime: str) -> bool:
        """
        Check if current regime favors scalping.

        Args:
            regime: Current regime label

        Returns:
            True if regime is favorable
        """
        favorable_regimes = ['high_volume', 'tight', 'trending_high_volume']
        return any(fav in regime for fav in favorable_regimes)

    def calculate_transaction_costs(
        self,
        signals: pd.Series,
        spread_pct: float = 0.001,
        commission: float = 0.0
    ) -> pd.Series:
        """
        Calculate estimated transaction costs for scalping.

        Args:
            signals: Signal series
            spread_pct: Bid-ask spread as percentage
            commission: Commission per trade

        Returns:
            Series with transaction costs
        """
        # Count trades (signal changes)
        signal_changes = signals.diff().abs()
        trades = signal_changes > 0

        # Calculate costs
        costs = pd.Series(0.0, index=signals.index)
        costs[trades] = spread_pct + commission

        return costs
