"""
Regime detection methods.
Implements 5 regime detectors: trend, volatility, volume, microstructure, composite.
"""
import numpy as np
import pandas as pd
from typing import Tuple
from .helpers import (
    calculate_adx,
    calculate_ema_slope,
    calculate_volume_zscore,
    calculate_volatility_regime
)


class TrendRegimeDetector:
    """Detects trending vs ranging regimes using ADX."""

    def __init__(self, adx_period: int = 14, adx_threshold: float = 25.0):
        """
        Initialize trend regime detector.

        Args:
            adx_period: ADX calculation period
            adx_threshold: Threshold for trending regime
        """
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold

    def detect(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect trend regime.

        Returns:
            Series with labels: 'trending', 'ranging'
        """
        adx = calculate_adx(df, self.adx_period)

        regime = pd.Series('ranging', index=df.index)
        regime[adx > self.adx_threshold] = 'trending'

        return regime

    def detect_numeric(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect trend regime with numeric labels.

        Returns:
            Series with labels: 1 = trending, 0 = ranging
        """
        adx = calculate_adx(df, self.adx_period)
        return (adx > self.adx_threshold).astype(int)


class VolatilityRegimeDetector:
    """Detects high vs low volatility regimes."""

    def __init__(self, window: int = 20, threshold: float = 1.0):
        """
        Initialize volatility regime detector.

        Args:
            window: Window for volatility calculation
            threshold: Z-score threshold
        """
        self.window = window
        self.threshold = threshold

    def detect(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect volatility regime.

        Returns:
            Series with labels: 'high_vol', 'normal_vol', 'low_vol'
        """
        returns = df['Close'].pct_change()
        regime_numeric = calculate_volatility_regime(returns, self.window, self.threshold)

        regime = pd.Series('normal_vol', index=df.index)
        regime[regime_numeric == 1] = 'high_vol'
        regime[regime_numeric == -1] = 'low_vol'

        return regime

    def detect_numeric(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect volatility regime with numeric labels.

        Returns:
            Series with labels: 1 = high, 0 = normal, -1 = low
        """
        returns = df['Close'].pct_change()
        return calculate_volatility_regime(returns, self.window, self.threshold)


class VolumeRegimeDetector:
    """Detects high vs low volume regimes."""

    def __init__(self, window: int = 20, threshold: float = 1.5):
        """
        Initialize volume regime detector.

        Args:
            window: Window for volume statistics
            threshold: Z-score threshold
        """
        self.window = window
        self.threshold = threshold

    def detect(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect volume regime.

        Returns:
            Series with labels: 'high_volume', 'normal_volume', 'low_volume'
        """
        vol_zscore = calculate_volume_zscore(df['Volume'], self.window)

        regime = pd.Series('normal_volume', index=df.index)
        regime[vol_zscore > self.threshold] = 'high_volume'
        regime[vol_zscore < -self.threshold] = 'low_volume'

        return regime

    def detect_numeric(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect volume regime with numeric labels.

        Returns:
            Series with labels: 1 = high, 0 = normal, -1 = low
        """
        vol_zscore = calculate_volume_zscore(df['Volume'], self.window)

        regime = pd.Series(0, index=df.index)
        regime[vol_zscore > self.threshold] = 1
        regime[vol_zscore < -self.threshold] = -1

        return regime


class MicrostructureRegimeDetector:
    """Detects regimes based on market microstructure."""

    def __init__(self, window: int = 20, spread_threshold: float = 1.5):
        """
        Initialize microstructure regime detector.

        Args:
            window: Window for calculations
            spread_threshold: Threshold for spread regime
        """
        self.window = window
        self.spread_threshold = spread_threshold

    def detect(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect microstructure regime.

        Returns:
            Series with labels: 'tight', 'wide'
        """
        # Use HL spread as proxy
        spread = (df['High'] - df['Low']) / df['Close']
        spread_ma = spread.rolling(self.window).mean()
        spread_std = spread.rolling(self.window).std()

        spread_zscore = (spread - spread_ma) / (spread_std + 1e-10)

        regime = pd.Series('tight', index=df.index)
        regime[spread_zscore > self.spread_threshold] = 'wide'

        return regime

    def detect_numeric(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect microstructure regime with numeric labels.

        Returns:
            Series with labels: 1 = wide, 0 = tight
        """
        spread = (df['High'] - df['Low']) / df['Close']
        spread_ma = spread.rolling(self.window).mean()
        spread_std = spread.rolling(self.window).std()

        spread_zscore = (spread - spread_ma) / (spread_std + 1e-10)

        return (spread_zscore > self.spread_threshold).astype(int)


class CompositeRegimeDetector:
    """Combines multiple regime detectors."""

    def __init__(
        self,
        use_trend: bool = True,
        use_volatility: bool = True,
        use_volume: bool = True,
        use_microstructure: bool = True
    ):
        """
        Initialize composite regime detector.

        Args:
            use_trend: Use trend detector
            use_volatility: Use volatility detector
            use_volume: Use volume detector
            use_microstructure: Use microstructure detector
        """
        self.use_trend = use_trend
        self.use_volatility = use_volatility
        self.use_volume = use_volume
        self.use_microstructure = use_microstructure

        # Initialize detectors
        self.trend_detector = TrendRegimeDetector()
        self.vol_detector = VolatilityRegimeDetector()
        self.volume_detector = VolumeRegimeDetector()
        self.micro_detector = MicrostructureRegimeDetector()

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect regimes using all enabled detectors.

        Returns:
            DataFrame with regime labels from each detector
        """
        regimes = pd.DataFrame(index=df.index)

        if self.use_trend:
            regimes['trend_regime'] = self.trend_detector.detect(df)

        if self.use_volatility:
            regimes['vol_regime'] = self.vol_detector.detect(df)

        if self.use_volume:
            regimes['volume_regime'] = self.volume_detector.detect(df)

        if self.use_microstructure:
            regimes['micro_regime'] = self.micro_detector.detect(df)

        return regimes

    def detect_composite_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Create composite regime by combining all detectors.

        Returns:
            Series with composite regime labels
        """
        regimes_df = self.detect(df)

        # Combine regimes into single label
        composite = pd.Series('', index=df.index)

        for idx in df.index:
            regime_parts = []

            if 'trend_regime' in regimes_df.columns:
                regime_parts.append(regimes_df.loc[idx, 'trend_regime'])

            if 'vol_regime' in regimes_df.columns:
                regime_parts.append(regimes_df.loc[idx, 'vol_regime'])

            composite.loc[idx] = '_'.join(regime_parts) if regime_parts else 'unknown'

        return composite

    def get_regime_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get numeric regime matrix for modeling.

        Returns:
            DataFrame with numeric regime indicators
        """
        regime_matrix = pd.DataFrame(index=df.index)

        if self.use_trend:
            regime_matrix['regime_trending'] = self.trend_detector.detect_numeric(df)

        if self.use_volatility:
            regime_matrix['regime_vol'] = self.vol_detector.detect_numeric(df)

        if self.use_volume:
            regime_matrix['regime_volume'] = self.volume_detector.detect_numeric(df)

        if self.use_microstructure:
            regime_matrix['regime_micro'] = self.micro_detector.detect_numeric(df)

        return regime_matrix


def detect_all_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to detect all regime types.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with all regime labels
    """
    detector = CompositeRegimeDetector(
        use_trend=True,
        use_volatility=True,
        use_volume=True,
        use_microstructure=True
    )

    return detector.detect(df)


def get_current_regime(
    df: pd.DataFrame,
    lookback: int = 1
) -> dict:
    """
    Get current regime state.

    Args:
        df: DataFrame with OHLCV data
        lookback: How many periods back (1 = most recent)

    Returns:
        Dictionary with current regime states
    """
    regimes = detect_all_regimes(df)

    current = {}
    for col in regimes.columns:
        current[col] = regimes[col].iloc[-lookback]

    return current
