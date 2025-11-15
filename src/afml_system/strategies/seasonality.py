"""
Seasonality strategy.
Exploits calendar and time-based patterns.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict


class SeasonalityStrategy:
    """Seasonality-based trading strategy."""

    def __init__(
        self,
        use_day_of_week: bool = True,
        use_month: bool = True,
        use_time_of_day: bool = False
    ):
        """
        Initialize seasonality strategy.

        Args:
            use_day_of_week: Use day of week patterns
            use_month: Use monthly patterns
            use_time_of_day: Use intraday patterns
        """
        self.use_day_of_week = use_day_of_week
        self.use_month = use_month
        self.use_time_of_day = use_time_of_day
        self.historical_patterns = {}

    def fit_patterns(self, df: pd.DataFrame):
        """
        Learn historical seasonal patterns.

        Args:
            df: DataFrame with OHLCV data and datetime index
        """
        returns = df['Close'].pct_change()

        # Day of week patterns
        if self.use_day_of_week:
            dow_returns = returns.groupby(returns.index.dayofweek).mean()
            self.historical_patterns['day_of_week'] = dow_returns.to_dict()

        # Monthly patterns
        if self.use_month:
            month_returns = returns.groupby(returns.index.month).mean()
            self.historical_patterns['month'] = month_returns.to_dict()

        # Hour patterns (if intraday data)
        if self.use_time_of_day and hasattr(returns.index, 'hour'):
            hour_returns = returns.groupby(returns.index.hour).mean()
            self.historical_patterns['hour'] = hour_returns.to_dict()

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate seasonality signals.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series with signals (-1, 0, 1)
        """
        signals = pd.Series(0, index=df.index)

        if not self.historical_patterns:
            return signals

        # Combine seasonal factors
        seasonal_score = pd.Series(0.0, index=df.index)

        # Day of week
        if 'day_of_week' in self.historical_patterns:
            for idx in df.index:
                dow = idx.dayofweek
                seasonal_score.loc[idx] += self.historical_patterns['day_of_week'].get(dow, 0)

        # Month
        if 'month' in self.historical_patterns:
            for idx in df.index:
                month = idx.month
                seasonal_score.loc[idx] += self.historical_patterns['month'].get(month, 0)

        # Generate signals based on seasonal score
        score_std = seasonal_score.std()
        if score_std > 0:
            signals[seasonal_score > score_std] = 1
            signals[seasonal_score < -score_std] = -1

        return signals

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate seasonality features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with features
        """
        features = pd.DataFrame(index=df.index)

        # Day of week (one-hot encoded)
        for dow in range(5):  # Monday to Friday
            features[f'dow_{dow}'] = (df.index.dayofweek == dow).astype(int)

        # Month (one-hot encoded)
        for month in range(1, 13):
            features[f'month_{month}'] = (df.index.month == month).astype(int)

        # Week of month
        features['week_of_month'] = (df.index.day - 1) // 7

        # Beginning/end of month
        features['is_month_start'] = (df.index.day <= 5).astype(int)
        features['is_month_end'] = (df.index.day >= 25).astype(int)

        # Quarter
        features['quarter'] = df.index.quarter

        # Historical seasonal returns (if patterns fitted)
        if self.historical_patterns:
            features['seasonal_dow_return'] = df.index.map(
                lambda x: self.historical_patterns.get('day_of_week', {}).get(x.dayofweek, 0)
            )
            features['seasonal_month_return'] = df.index.map(
                lambda x: self.historical_patterns.get('month', {}).get(x.month, 0)
            )

        # Recent performance by day of week
        returns = df['Close'].pct_change()
        for dow in range(5):
            dow_mask = df.index.dayofweek == dow
            features[f'recent_dow_{dow}_return'] = returns[dow_mask].rolling(4, min_periods=1).mean().reindex(df.index).fillna(0)

        return features

    def get_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate signal strength (0 to 1).

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series with signal strengths
        """
        if not self.historical_patterns:
            return pd.Series(0.0, index=df.index)

        # Calculate seasonal score
        seasonal_score = pd.Series(0.0, index=df.index)

        if 'day_of_week' in self.historical_patterns:
            for idx in df.index:
                dow = idx.dayofweek
                seasonal_score.loc[idx] += abs(self.historical_patterns['day_of_week'].get(dow, 0))

        # Normalize
        max_score = seasonal_score.max()
        if max_score > 0:
            strength = seasonal_score / max_score
        else:
            strength = pd.Series(0.0, index=df.index)

        return strength.clip(0, 1)

    def is_favorable_regime(self, regime: str) -> bool:
        """
        Check if current regime favors seasonality strategy.

        Args:
            regime: Current regime label

        Returns:
            True if regime is favorable (seasonality works in all regimes)
        """
        # Seasonality can work in any regime
        return True

    def get_calendar_anomalies(self, df: pd.DataFrame) -> Dict:
        """
        Identify calendar anomalies in data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dictionary with anomaly statistics
        """
        returns = df['Close'].pct_change()

        anomalies = {}

        # Monday effect
        monday_returns = returns[returns.index.dayofweek == 0]
        other_returns = returns[returns.index.dayofweek != 0]
        anomalies['monday_effect'] = monday_returns.mean() - other_returns.mean()

        # Month-end effect
        month_end = returns[returns.index.day >= 25]
        month_middle = returns[(returns.index.day >= 10) & (returns.index.day <= 20)]
        anomalies['month_end_effect'] = month_end.mean() - month_middle.mean()

        # January effect
        january = returns[returns.index.month == 1]
        other_months = returns[returns.index.month != 1]
        anomalies['january_effect'] = january.mean() - other_months.mean()

        return anomalies
