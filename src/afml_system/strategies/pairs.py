"""
Pairs trading strategy.
Statistical arbitrage between correlated assets.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple


class PairsStrategy:
    """Pairs trading strategy."""

    def __init__(
        self,
        window: int = 60,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5
    ):
        """
        Initialize pairs strategy.

        Args:
            window: Window for calculating spread statistics
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
        """
        self.window = window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def calculate_spread(
        self,
        price1: pd.Series,
        price2: pd.Series,
        use_log: bool = True
    ) -> pd.Series:
        """
        Calculate spread between two price series.

        Args:
            price1: First price series
            price2: Second price series
            use_log: Use log prices

        Returns:
            Spread series
        """
        if use_log:
            spread = np.log(price1) - np.log(price2)
        else:
            # Use hedge ratio
            hedge_ratio = self._calculate_hedge_ratio(price1, price2)
            spread = price1 - hedge_ratio * price2

        return spread

    def _calculate_hedge_ratio(
        self,
        price1: pd.Series,
        price2: pd.Series,
        window: Optional[int] = None
    ) -> float:
        """Calculate hedge ratio between two series."""
        if window is None:
            window = self.window

        # Use OLS regression
        recent_p1 = price1.iloc[-window:]
        recent_p2 = price2.iloc[-window:]

        # Simple ratio
        hedge_ratio = (recent_p1 * recent_p2).sum() / (recent_p2 ** 2).sum()

        return hedge_ratio

    def generate_signals(
        self,
        price1: pd.Series,
        price2: pd.Series
    ) -> pd.Series:
        """
        Generate pairs trading signals.

        Args:
            price1: First price series
            price2: Second price series

        Returns:
            Series with signals (-1, 0, 1)
        """
        # Calculate spread
        spread = self.calculate_spread(price1, price2)

        # Calculate z-score
        spread_ma = spread.rolling(self.window).mean()
        spread_std = spread.rolling(self.window).std()
        zscore = (spread - spread_ma) / (spread_std + 1e-10)

        # Generate signals
        signals = pd.Series(0, index=spread.index)

        # Entry signals
        signals[zscore > self.entry_threshold] = -1  # Spread too high - short
        signals[zscore < -self.entry_threshold] = 1  # Spread too low - long

        # Exit signals (mean reversion)
        signals[abs(zscore) < self.exit_threshold] = 0

        return signals

    def calculate_features(
        self,
        price1: pd.Series,
        price2: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate pairs trading features.

        Args:
            price1: First price series
            price2: Second price series

        Returns:
            DataFrame with features
        """
        features = pd.DataFrame(index=price1.index)

        # Spread
        spread = self.calculate_spread(price1, price2)
        features['spread'] = spread

        # Z-score
        spread_ma = spread.rolling(self.window).mean()
        spread_std = spread.rolling(self.window).std()
        features['spread_zscore'] = (spread - spread_ma) / (spread_std + 1e-10)

        # Correlation
        for window in [20, 60, 120]:
            features[f'correlation_{window}'] = price1.rolling(window).corr(price2)

        # Spread volatility
        features['spread_vol'] = spread.rolling(20).std()

        # Half-life of mean reversion
        features['half_life'] = self._calculate_half_life(spread)

        # Cointegration score (simplified)
        features['coint_score'] = self._cointegration_score(price1, price2)

        # Hedge ratio
        features['hedge_ratio'] = self._rolling_hedge_ratio(price1, price2)

        return features

    def _calculate_half_life(self, spread: pd.Series, window: int = 60) -> pd.Series:
        """Calculate half-life of mean reversion."""
        half_lives = pd.Series(index=spread.index, dtype=float)

        for i in range(window, len(spread)):
            window_spread = spread.iloc[i-window:i]

            # Simple AR(1) model
            spread_lag = window_spread.shift()
            spread_diff = window_spread.diff()

            # Remove NaN
            valid = ~(spread_lag.isna() | spread_diff.isna())
            if valid.sum() > 10:
                # Beta coefficient
                beta = (spread_diff[valid] * spread_lag[valid]).sum() / (spread_lag[valid] ** 2).sum()

                if beta < 0:
                    half_life = -np.log(2) / beta
                else:
                    half_life = np.inf

                half_lives.iloc[i] = half_life
            else:
                half_lives.iloc[i] = np.nan

        return half_lives

    def _cointegration_score(
        self,
        price1: pd.Series,
        price2: pd.Series,
        window: int = 100
    ) -> pd.Series:
        """Calculate rolling cointegration score."""
        scores = pd.Series(index=price1.index, dtype=float)

        for i in range(window, len(price1)):
            p1_window = price1.iloc[i-window:i]
            p2_window = price2.iloc[i-window:i]

            # Calculate spread stationarity (simplified)
            spread = np.log(p1_window) - np.log(p2_window)
            spread_change = spread.diff()

            # Score based on mean reversion
            score = abs(spread_change.autocorr(1)) if len(spread_change.dropna()) > 1 else 0
            scores.iloc[i] = score

        return scores

    def _rolling_hedge_ratio(
        self,
        price1: pd.Series,
        price2: pd.Series
    ) -> pd.Series:
        """Calculate rolling hedge ratio."""
        hedge_ratios = pd.Series(index=price1.index, dtype=float)

        for i in range(self.window, len(price1)):
            p1_window = price1.iloc[i-self.window:i]
            p2_window = price2.iloc[i-self.window:i]

            ratio = (p1_window * p2_window).sum() / (p2_window ** 2).sum()
            hedge_ratios.iloc[i] = ratio

        return hedge_ratios

    def get_signal_strength(
        self,
        price1: pd.Series,
        price2: pd.Series
    ) -> pd.Series:
        """
        Calculate signal strength (0 to 1).

        Args:
            price1: First price series
            price2: Second price series

        Returns:
            Series with signal strengths
        """
        spread = self.calculate_spread(price1, price2)
        spread_ma = spread.rolling(self.window).mean()
        spread_std = spread.rolling(self.window).std()
        zscore = abs((spread - spread_ma) / (spread_std + 1e-10))

        strength = zscore / self.entry_threshold

        return strength.clip(0, 1)

    def is_favorable_regime(self, regime: str) -> bool:
        """
        Check if current regime favors pairs trading.

        Args:
            regime: Current regime label

        Returns:
            True if regime is favorable
        """
        favorable_regimes = ['ranging', 'low_vol', 'normal_vol']
        return any(fav in regime for fav in favorable_regimes)
