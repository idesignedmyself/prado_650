"""
Sentiment strategy.
Uses market sentiment indicators and proxy measures.
"""
import numpy as np
import pandas as pd
from typing import Optional


class SentimentStrategy:
    """Sentiment-based trading strategy."""

    def __init__(
        self,
        sentiment_window: int = 20,
        threshold: float = 1.5
    ):
        """
        Initialize sentiment strategy.

        Args:
            sentiment_window: Window for sentiment aggregation
            threshold: Threshold for sentiment extremes
        """
        self.sentiment_window = sentiment_window
        self.threshold = threshold

    def calculate_sentiment_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate market sentiment score from price action.

        Uses multiple proxies for sentiment:
        - Up/down volume ratio
        - Price momentum
        - Volatility changes
        - New highs/lows

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Sentiment score series
        """
        sentiment = pd.Series(0.0, index=df.index)

        # 1. Volume sentiment (up vs down volume)
        price_change = df['Close'].diff()
        up_volume = df['Volume'].copy()
        down_volume = df['Volume'].copy()

        up_volume[price_change <= 0] = 0
        down_volume[price_change >= 0] = 0

        up_vol_sum = up_volume.rolling(self.sentiment_window).sum()
        down_vol_sum = down_volume.rolling(self.sentiment_window).sum()

        volume_sentiment = (up_vol_sum - down_vol_sum) / (up_vol_sum + down_vol_sum + 1e-10)

        # 2. Momentum sentiment
        momentum = df['Close'].pct_change(self.sentiment_window)
        momentum_zscore = (momentum - momentum.rolling(50).mean()) / (momentum.rolling(50).std() + 1e-10)

        # 3. Volatility sentiment (increasing vol = fear)
        returns = df['Close'].pct_change()
        vol = returns.rolling(self.sentiment_window).std()
        vol_change = vol.pct_change(self.sentiment_window)
        vol_sentiment = -vol_change  # Negative because high vol = negative sentiment

        # 4. New high/low sentiment
        high_20 = df['High'].rolling(self.sentiment_window).max()
        low_20 = df['Low'].rolling(self.sentiment_window).min()

        near_high = (df['Close'] - low_20) / (high_20 - low_20 + 1e-10)
        high_low_sentiment = (near_high - 0.5) * 2  # Scale to [-1, 1]

        # Combine sentiments
        sentiment = (
            0.3 * volume_sentiment +
            0.3 * momentum_zscore +
            0.2 * vol_sentiment +
            0.2 * high_low_sentiment
        )

        return sentiment

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate sentiment signals.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series with signals (-1, 0, 1)
        """
        sentiment = self.calculate_sentiment_score(df)

        # Normalize sentiment
        sentiment_ma = sentiment.rolling(self.sentiment_window * 2).mean()
        sentiment_std = sentiment.rolling(self.sentiment_window * 2).std()
        sentiment_zscore = (sentiment - sentiment_ma) / (sentiment_std + 1e-10)

        signals = pd.Series(0, index=df.index)

        # Contrarian signals (fade extreme sentiment)
        signals[sentiment_zscore > self.threshold] = -1  # Extreme optimism - sell
        signals[sentiment_zscore < -self.threshold] = 1  # Extreme pessimism - buy

        return signals

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sentiment features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with features
        """
        features = pd.DataFrame(index=df.index)

        # Overall sentiment score
        features['sentiment_score'] = self.calculate_sentiment_score(df)

        # Volume-based sentiment
        price_change = df['Close'].diff()
        up_volume = df['Volume'].where(price_change > 0, 0)
        down_volume = df['Volume'].where(price_change < 0, 0)

        for window in [5, 10, 20]:
            up_sum = up_volume.rolling(window).sum()
            down_sum = down_volume.rolling(window).sum()
            features[f'volume_sentiment_{window}'] = (up_sum - down_sum) / (up_sum + down_sum + 1e-10)

        # Put-call ratio proxy (using volume and price action)
        features['put_call_proxy'] = self._calculate_put_call_proxy(df)

        # Fear index proxy (volatility changes)
        returns = df['Close'].pct_change()
        vol = returns.rolling(20).std()
        features['fear_index'] = vol / vol.rolling(50).mean()

        # Advance-decline proxy
        features['advance_decline'] = self._calculate_advance_decline(df)

        # Sentiment extremes
        sentiment = features['sentiment_score']
        for window in [20, 50]:
            features[f'sentiment_percentile_{window}'] = sentiment.rolling(window).rank(pct=True)

        # Sentiment changes
        features['sentiment_change'] = sentiment.diff()
        features['sentiment_acceleration'] = sentiment.diff().diff()

        # Bull-bear spread
        features['bull_bear_spread'] = self._calculate_bull_bear_spread(df)

        return features

    def _calculate_put_call_proxy(self, df: pd.DataFrame) -> pd.Series:
        """Calculate put-call ratio proxy from volume."""
        # Use down volume / up volume as proxy
        price_change = df['Close'].diff()
        up_volume = df['Volume'].where(price_change > 0, 0)
        down_volume = df['Volume'].where(price_change < 0, 0)

        put_call = down_volume.rolling(20).sum() / (up_volume.rolling(20).sum() + 1e-10)

        return put_call

    def _calculate_advance_decline(self, df: pd.DataFrame) -> pd.Series:
        """Calculate advance-decline indicator."""
        # Simplified: cumulative sum of up/down days
        price_change = df['Close'].diff()
        advances = (price_change > 0).astype(int)
        declines = (price_change < 0).astype(int)

        ad_line = (advances - declines).cumsum()

        return ad_line

    def _calculate_bull_bear_spread(self, df: pd.DataFrame) -> pd.Series:
        """Calculate bull-bear spread."""
        # Percentage of recent days that were up
        returns = df['Close'].pct_change()
        bull_pct = (returns > 0).rolling(20).mean()
        bear_pct = (returns < 0).rolling(20).mean()

        spread = bull_pct - bear_pct

        return spread

    def get_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate signal strength (0 to 1).

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series with signal strengths
        """
        sentiment = self.calculate_sentiment_score(df)
        sentiment_abs = abs(sentiment)

        # Normalize
        sentiment_max = sentiment_abs.rolling(50).max()
        strength = sentiment_abs / (sentiment_max + 1e-10)

        return strength.clip(0, 1)

    def is_favorable_regime(self, regime: str) -> bool:
        """
        Check if current regime favors sentiment strategy.

        Args:
            regime: Current regime label

        Returns:
            True if regime is favorable
        """
        # Sentiment works best in extreme regimes
        favorable_regimes = ['high_vol', 'trending', 'ranging']
        return any(fav in regime for fav in favorable_regimes)

    def get_sentiment_extremes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify sentiment extremes.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with extreme sentiment periods
        """
        sentiment = self.calculate_sentiment_score(df)
        sentiment_ma = sentiment.rolling(50).mean()
        sentiment_std = sentiment.rolling(50).std()
        sentiment_zscore = (sentiment - sentiment_ma) / (sentiment_std + 1e-10)

        extremes = pd.DataFrame(index=df.index)
        extremes['sentiment'] = sentiment
        extremes['zscore'] = sentiment_zscore
        extremes['is_extreme'] = abs(sentiment_zscore) > self.threshold
        extremes['extreme_type'] = 'neutral'
        extremes.loc[sentiment_zscore > self.threshold, 'extreme_type'] = 'extreme_bullish'
        extremes.loc[sentiment_zscore < -self.threshold, 'extreme_type'] = 'extreme_bearish'

        return extremes
