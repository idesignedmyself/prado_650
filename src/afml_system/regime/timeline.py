"""
Regime timeline builder.
Creates temporal regime timelines and transitions.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime


class RegimeTimeline:
    """Builds and manages regime timelines."""

    def __init__(self, regime_series: pd.Series, regime_name: str = "regime"):
        """
        Initialize regime timeline.

        Args:
            regime_series: Series with regime labels
            regime_name: Name of the regime type
        """
        self.regime_series = regime_series
        self.regime_name = regime_name
        self.transitions = self._identify_transitions()
        self.periods = self._identify_periods()

    def _identify_transitions(self) -> pd.DataFrame:
        """Identify regime transition points."""
        transitions = []

        prev_regime = None
        for idx, regime in self.regime_series.items():
            if prev_regime is not None and regime != prev_regime:
                transitions.append({
                    'timestamp': idx,
                    'from_regime': prev_regime,
                    'to_regime': regime
                })
            prev_regime = regime

        return pd.DataFrame(transitions)

    def _identify_periods(self) -> List[Dict]:
        """Identify continuous regime periods."""
        periods = []

        start_idx = None
        current_regime = None

        for i, (idx, regime) in enumerate(self.regime_series.items()):
            if current_regime is None:
                # First period
                start_idx = idx
                current_regime = regime
            elif regime != current_regime:
                # Regime changed - save previous period
                periods.append({
                    'regime': current_regime,
                    'start': start_idx,
                    'end': self.regime_series.index[i-1],
                    'duration': i - self.regime_series.index.get_loc(start_idx)
                })
                start_idx = idx
                current_regime = regime

        # Add final period
        if current_regime is not None:
            periods.append({
                'regime': current_regime,
                'start': start_idx,
                'end': self.regime_series.index[-1],
                'duration': len(self.regime_series) - self.regime_series.index.get_loc(start_idx)
            })

        return periods

    def get_regime_at(self, timestamp) -> Optional[str]:
        """Get regime at specific timestamp."""
        if timestamp in self.regime_series.index:
            return self.regime_series.loc[timestamp]
        return None

    def get_current_regime(self) -> str:
        """Get most recent regime."""
        return self.regime_series.iloc[-1]

    def get_regime_duration(self, timestamp) -> int:
        """Get duration of current regime at timestamp."""
        current_regime = self.get_regime_at(timestamp)
        if current_regime is None:
            return 0

        # Count consecutive periods
        loc = self.regime_series.index.get_loc(timestamp)
        duration = 1

        # Look back
        for i in range(loc - 1, -1, -1):
            if self.regime_series.iloc[i] == current_regime:
                duration += 1
            else:
                break

        return duration

    def get_transition_count(self) -> int:
        """Get total number of transitions."""
        return len(self.transitions)

    def get_regime_statistics(self) -> pd.DataFrame:
        """Get statistics for each regime."""
        stats = []

        for regime_label in self.regime_series.unique():
            regime_periods = [p for p in self.periods if p['regime'] == regime_label]

            if regime_periods:
                durations = [p['duration'] for p in regime_periods]

                stats.append({
                    'regime': regime_label,
                    'count': len(regime_periods),
                    'total_duration': sum(durations),
                    'avg_duration': np.mean(durations),
                    'max_duration': max(durations),
                    'min_duration': min(durations)
                })

        return pd.DataFrame(stats)

    def plot_timeline(self) -> pd.DataFrame:
        """
        Create timeline visualization data.

        Returns:
            DataFrame suitable for plotting
        """
        timeline_data = []

        for period in self.periods:
            timeline_data.append({
                'start': period['start'],
                'end': period['end'],
                'regime': period['regime'],
                'duration': period['duration']
            })

        return pd.DataFrame(timeline_data)


class MultiRegimeTimeline:
    """Manages multiple regime timelines simultaneously."""

    def __init__(self, regime_df: pd.DataFrame):
        """
        Initialize multi-regime timeline.

        Args:
            regime_df: DataFrame with multiple regime columns
        """
        self.regime_df = regime_df
        self.timelines = {}

        # Create timeline for each regime type
        for col in regime_df.columns:
            self.timelines[col] = RegimeTimeline(regime_df[col], col)

    def get_composite_regime(self, timestamp) -> Dict[str, str]:
        """Get all regime states at timestamp."""
        regimes = {}
        for name, timeline in self.timelines.items():
            regimes[name] = timeline.get_regime_at(timestamp)
        return regimes

    def get_regime_combination_frequency(self) -> pd.DataFrame:
        """Get frequency of regime combinations."""
        # Create composite regime label
        composite = self.regime_df.apply(
            lambda row: '_'.join(row.astype(str)),
            axis=1
        )

        # Count frequencies
        freq = composite.value_counts()

        return pd.DataFrame({
            'combination': freq.index,
            'count': freq.values,
            'frequency': freq.values / len(composite)
        })

    def get_all_statistics(self) -> Dict[str, pd.DataFrame]:
        """Get statistics for all regime types."""
        stats = {}
        for name, timeline in self.timelines.items():
            stats[name] = timeline.get_regime_statistics()
        return stats

    def find_stable_periods(self, min_duration: int = 10) -> pd.DataFrame:
        """
        Find periods where all regimes are stable.

        Args:
            min_duration: Minimum duration to consider

        Returns:
            DataFrame with stable periods
        """
        stable_periods = []

        # Check each timestamp
        for i in range(len(self.regime_df)):
            timestamp = self.regime_df.index[i]

            # Get durations for all regimes
            durations = {}
            for name, timeline in self.timelines.items():
                durations[name] = timeline.get_regime_duration(timestamp)

            # Check if all durations exceed threshold
            if all(d >= min_duration for d in durations.values()):
                regimes = self.get_composite_regime(timestamp)
                stable_periods.append({
                    'timestamp': timestamp,
                    **regimes,
                    'min_duration': min(durations.values())
                })

        return pd.DataFrame(stable_periods)


def build_regime_timeline(
    df: pd.DataFrame,
    regime_detectors: Optional[dict] = None
) -> MultiRegimeTimeline:
    """
    Build complete regime timeline from data.

    Args:
        df: DataFrame with OHLCV data
        regime_detectors: Optional dict of regime detectors

    Returns:
        MultiRegimeTimeline object
    """
    from .detection import CompositeRegimeDetector

    if regime_detectors is None:
        # Use default composite detector
        detector = CompositeRegimeDetector()
        regime_df = detector.detect(df)
    else:
        # Use custom detectors
        regime_df = pd.DataFrame(index=df.index)
        for name, detector in regime_detectors.items():
            regime_df[name] = detector.detect(df)

    return MultiRegimeTimeline(regime_df)


def get_regime_transitions(
    regime_series: pd.Series,
    include_duration: bool = True
) -> pd.DataFrame:
    """
    Get regime transition information.

    Args:
        regime_series: Series with regime labels
        include_duration: Include duration in each regime

    Returns:
        DataFrame with transition information
    """
    transitions = []
    prev_regime = None
    prev_timestamp = None
    duration = 0

    for timestamp, regime in regime_series.items():
        if prev_regime is not None and regime != prev_regime:
            transition = {
                'timestamp': timestamp,
                'from': prev_regime,
                'to': regime
            }

            if include_duration:
                transition['duration_in_previous'] = duration

            transitions.append(transition)
            duration = 1
        else:
            duration += 1

        prev_regime = regime
        prev_timestamp = timestamp

    return pd.DataFrame(transitions)


def analyze_regime_persistence(
    regime_series: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Analyze how persistent regimes are.

    Args:
        regime_series: Series with regime labels
        window: Window for analysis

    Returns:
        Series with persistence scores
    """
    persistence = pd.Series(index=regime_series.index, dtype=float)

    for i in range(len(regime_series)):
        if i < window:
            persistence.iloc[i] = np.nan
        else:
            window_regimes = regime_series.iloc[i-window:i]
            current = regime_series.iloc[i]
            persistence.iloc[i] = (window_regimes == current).sum() / window

    return persistence
