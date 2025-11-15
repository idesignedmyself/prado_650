"""
Volatility strategy.
Trades based on volatility regimes and changes.
"""
import numpy as np
import pandas as pd
from typing import Optional


class VolatilityStrategy:
    """Volatility-based trading strategy."""

    def __init__(
        self,
        window: int = 20,
        vol_threshold: float = 0.02,
        use_vol_breakout: bool = True
    ):
        """
        Initialize volatility strategy.

        Args:
            window: Window for volatility calculation
            vol_threshold: Volatility threshold
            use_vol_breakout: Trade volatility breakouts
        """
        self.window = window
        self.vol_threshold = vol_threshold
        self.use_vol_breakout = use_vol_breakout

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate volatility signals.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series with signals (-1, 0, 1)
        """
        # Calculate volatility
        returns = df['Close'].pct_change()
        vol = returns.rolling(self.window).std()

        # Volatility regime
        vol_ma = vol.rolling(self.window * 2).mean()
        vol_std = vol.rolling(self.window * 2).std()

        signals = pd.Series(0, index=df.index)

        if self.use_vol_breakout:
            # Trade on volatility breakouts
            vol_breakout = vol > (vol_ma + vol_std)
            price_direction = returns.rolling(5).mean()

            # Enter in direction of price move during vol breakout
            signals[vol_breakout & (price_direction > 0)] = 1
            signals[vol_breakout & (price_direction < 0)] = -1

        else:
            # Trade on volatility compression
            vol_compressed = vol < (vol_ma - vol_std)

            # Anticipate expansion - enter in trend direction
            price_trend = df['Close'].rolling(20).mean()
            signals[vol_compressed & (df['Close'] > price_trend)] = 1
            signals[vol_compressed & (df['Close'] < price_trend)] = -1

        return signals

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with features
        """
        features = pd.DataFrame(index=df.index)

        returns = df['Close'].pct_change()

        # Realized volatility at multiple windows
        for window in [5, 10, 20, 50]:
            features[f'vol_realized_{window}'] = returns.rolling(window).std()

        # Parkinson volatility
        hl = np.log(df['High'] / df['Low'])
        features['vol_parkinson'] = hl.rolling(self.window).std()

        # Garman-Klass volatility
        hl = np.log(df['High'] / df['Low'])
        co = np.log(df['Close'] / df['Open'])
        gk = 0.5 * hl**2 - (2*np.log(2)-1) * co**2
        features['vol_garman_klass'] = np.sqrt(gk.rolling(self.window).mean())

        # Volatility regime
        vol = returns.rolling(20).std()
        vol_ma = vol.rolling(50).mean()
        features['vol_regime'] = (vol / vol_ma).fillna(1.0)

        # Volatility change
        features['vol_change'] = vol.pct_change()

        # ATR
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift())
        tr3 = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        features['atr'] = tr.rolling(14).mean()
        features['atr_ratio'] = features['atr'] / df['Close']

        # Volatility percentile
        features['vol_percentile'] = vol.rolling(100).rank(pct=True)

        # Intraday range
        features['intraday_range'] = (df['High'] - df['Low']) / df['Close']

        return features

    def get_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate signal strength (0 to 1).

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series with signal strengths
        """
        returns = df['Close'].pct_change()
        vol = returns.rolling(self.window).std()
        vol_ma = vol.rolling(self.window * 2).mean()

        # Strength based on volatility deviation
        vol_deviation = abs(vol - vol_ma) / (vol_ma + 1e-10)

        strength = vol_deviation / (vol_deviation.rolling(50).max() + 1e-10)

        return strength.clip(0, 1)

    def is_favorable_regime(self, regime: str) -> bool:
        """
        Check if current regime favors volatility strategy.

        Args:
            regime: Current regime label

        Returns:
            True if regime is favorable
        """
        favorable_regimes = ['high_vol', 'trending_high_vol']
        return any(fav in regime for fav in favorable_regimes)
