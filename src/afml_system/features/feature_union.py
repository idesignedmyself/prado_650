"""
Feature union: combines all feature types into complete matrix.
Creates the 19-feature matrix for PRADO9.
"""
import numpy as np
import pandas as pd
from typing import Optional, List
from .stationarity import frac_diff, log_returns
from .volatility import get_all_volatility_estimates
from .microstructure import get_microstructure_features
from .technical import get_technical_features


def build_feature_matrix(
    df: pd.DataFrame,
    events: Optional[pd.DatetimeIndex] = None,
    include_microstructure: bool = True,
    include_technical: bool = True,
    include_volatility: bool = True,
    frac_diff_d: float = 0.4,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Build complete feature matrix for PRADO9 at event timestamps.

    Creates 19 core features:
    1. Fractionally differentiated price
    2-7. Six volatility estimators
    8-10. Microstructure (OFI, VPIN, Kyle's lambda)
    11-19. Technical indicators

    Args:
        df: DataFrame with OHLCV data
        events: Event timestamps (CUSUM-filtered). If None, uses df.index
        include_microstructure: Include microstructure features
        include_technical: Include technical features
        include_volatility: Include volatility features
        frac_diff_d: Fractional differentiation order

    Returns:
        DataFrame with complete feature matrix at event timestamps
    """
    # Build features on full dataset first
    features = pd.DataFrame(index=df.index)

    # 1. Fractionally differentiated price (stationary)
    features['frac_diff_price'] = frac_diff(df['Close'], d=frac_diff_d)

    # 2-7. Volatility estimators
    if include_volatility:
        vol_features = get_all_volatility_estimates(df, window=20)
        features['vol_parkinson'] = vol_features['parkinson']
        features['vol_garman_klass'] = vol_features['garman_klass']
        features['vol_rogers_satchell'] = vol_features['rogers_satchell']
        features['vol_yang_zhang'] = vol_features['yang_zhang']
        features['vol_ewma'] = vol_features['ewma']
        features['vol_realized'] = vol_features['realized']

    # 8-10. Microstructure features
    if include_microstructure:
        features['ofi_20'] = _safe_microstructure_feature(df, 'ofi')
        features['vpin'] = _safe_microstructure_feature(df, 'vpin')
        features['kyle_lambda'] = _safe_microstructure_feature(df, 'kyle')

    # 11-19. Technical indicators (9 features)
    if include_technical:
        features['rsi_14'] = _rsi(df['Close'], 14)
        features['macd'] = _macd_line(df['Close'])
        features['bb_position'] = _bb_position(df['Close'])
        features['atr_14'] = _atr(df, 14)
        features['adx_14'] = _adx(df)
        features['momentum_20'] = df['Close'].pct_change(20)
        features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        features['price_sma_ratio_20'] = df['Close'] / df['Close'].rolling(20).mean()
        features['stoch_k'] = _stochastic_k(df)

    # Remove NaN rows - use threshold to keep rows with some valid features
    # Instead of dropna() which removes ALL rows with ANY NaN,
    # use forward fill for NaN values (common for rolling window features)
    features = features.ffill().bfill()

    # Only drop rows where ALL features are NaN
    features = features.dropna(how='all')

    # If events specified, extract features only at event timestamps
    if events is not None:
        # Get overlap between features and events
        valid_events = events.intersection(features.index)

        if len(valid_events) == 0:
            raise ValueError(
                f"No overlap between events ({len(events)}) and features ({len(features)}). "
                f"Events range: {events[0]} to {events[-1]}, "
                f"Features range: {features.index[0]} to {features.index[-1]}"
            )

        features = features.loc[valid_events]
        if verbose:
            print(f"  Features extracted at {len(features)} event timestamps")

    return features


def build_extended_features(
    df: pd.DataFrame,
    windows: List[int] = [5, 10, 20, 50]
) -> pd.DataFrame:
    """
    Build extended feature set with multiple windows.

    Args:
        df: DataFrame with OHLCV data
        windows: List of windows for indicators

    Returns:
        Extended feature matrix
    """
    features = pd.DataFrame(index=df.index)

    # Stationary price features
    features['log_returns'] = log_returns(df['Close'])
    features['frac_diff_0.4'] = frac_diff(df['Close'], d=0.4)
    features['frac_diff_0.6'] = frac_diff(df['Close'], d=0.6)

    # Volatility
    vol_features = get_all_volatility_estimates(df, window=20)
    for col in vol_features.columns:
        features[f'vol_{col}'] = vol_features[col]

    # Microstructure
    micro_features = get_microstructure_features(df, windows=[10, 20])
    for col in micro_features.columns:
        features[f'micro_{col}'] = micro_features[col]

    # Technical
    tech_features = get_technical_features(df, windows=windows)
    for col in tech_features.columns:
        features[f'tech_{col}'] = tech_features[col]

    # Price patterns
    features['hl_range'] = (df['High'] - df['Low']) / df['Close']
    features['oc_range'] = abs(df['Close'] - df['Open']) / df['Open']

    for window in windows:
        features[f'returns_{window}'] = df['Close'].pct_change(window)
        features[f'volatility_{window}'] = df['Close'].pct_change().rolling(window).std()

    return features


def select_top_features(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 19,
    method: str = 'mutual_info'
) -> List[str]:
    """
    Select top N features using feature importance.

    Args:
        X: Feature matrix
        y: Target labels
        n_features: Number of features to select
        method: Selection method ('mutual_info', 'f_classif', 'random_forest')

    Returns:
        List of selected feature names
    """
    from sklearn.feature_selection import mutual_info_classif, f_classif, SelectKBest
    from sklearn.ensemble import RandomForestClassifier

    # Align X and y
    idx = X.index.intersection(y.index)
    X_aligned = X.loc[idx]
    y_aligned = y.loc[idx]

    if method == 'mutual_info':
        scores = mutual_info_classif(X_aligned, y_aligned)
        feature_scores = pd.Series(scores, index=X.columns)

    elif method == 'f_classif':
        scores, _ = f_classif(X_aligned, y_aligned)
        feature_scores = pd.Series(scores, index=X.columns)

    elif method == 'random_forest':
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_aligned, y_aligned)
        feature_scores = pd.Series(rf.feature_importances_, index=X.columns)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Select top features
    top_features = feature_scores.nlargest(n_features).index.tolist()

    return top_features


# Helper functions for individual features

def _safe_microstructure_feature(df: pd.DataFrame, feature_type: str) -> pd.Series:
    """Safely calculate microstructure feature."""
    try:
        from ..data.microstructure import order_flow_imbalance, vpin, kyle_lambda

        if feature_type == 'ofi':
            return order_flow_imbalance(df, window=20)
        elif feature_type == 'vpin':
            return vpin(df, n_buckets=50, window=20)
        elif feature_type == 'kyle':
            return kyle_lambda(df, window=20)
    except:
        return pd.Series(0, index=df.index)


def _rsi(series: pd.Series, window: int) -> pd.Series:
    """Calculate RSI."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _macd_line(series: pd.Series) -> pd.Series:
    """Calculate MACD line."""
    ema_12 = series.ewm(span=12, adjust=False).mean()
    ema_26 = series.ewm(span=26, adjust=False).mean()
    return ema_12 - ema_26


def _bb_position(series: pd.Series, window: int = 20) -> pd.Series:
    """Calculate Bollinger Band position."""
    middle = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = middle + 2 * std
    lower = middle - 2 * std
    return (series - lower) / (upper - lower + 1e-10)


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    """Calculate ATR."""
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift())
    tr3 = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def _adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate ADX."""
    from .technical import adx
    adx_df = adx(df, window)
    return adx_df['adx']


def _stochastic_k(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate Stochastic %K."""
    lowest_low = df['Low'].rolling(window).min()
    highest_high = df['High'].rolling(window).max()
    return 100 * (df['Close'] - lowest_low) / (highest_high - lowest_low + 1e-10)


def normalize_features(features: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
    """
    Normalize feature matrix.

    Args:
        features: Feature matrix
        method: Normalization method ('zscore', 'minmax', 'robust')

    Returns:
        Normalized features
    """
    if method == 'zscore':
        return (features - features.mean()) / (features.std() + 1e-10)

    elif method == 'minmax':
        return (features - features.min()) / (features.max() - features.min() + 1e-10)

    elif method == 'robust':
        median = features.median()
        iqr = features.quantile(0.75) - features.quantile(0.25)
        return (features - median) / (iqr + 1e-10)

    else:
        raise ValueError(f"Unknown normalization method: {method}")
