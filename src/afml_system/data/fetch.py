"""
Data fetching module using yfinance.
Provides functions to download and prepare market data.
"""
import pandas as pd
import yfinance as yf
from typing import List, Optional, Union
from datetime import datetime, timedelta


def fetch_ohlcv(
    symbols: Union[str, List[str]],
    start_date: str,
    end_date: str,
    interval: str = "1d",
    progress: bool = True
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance.

    Args:
        symbols: Symbol or list of symbols to fetch
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)
        progress: Show download progress

    Returns:
        DataFrame with OHLCV data
    """
    if isinstance(symbols, str):
        symbols = [symbols]

    # Download data
    data = yf.download(
        symbols,
        start=start_date,
        end=end_date,
        interval=interval,
        progress=progress,
        group_by='ticker' if len(symbols) > 1 else None
    )

    if data.empty:
        raise ValueError(f"No data retrieved for {symbols}")

    # Handle single vs multiple symbols
    if len(symbols) == 1:
        data = data.copy()
        data.columns = pd.MultiIndex.from_product([[symbols[0]], data.columns])

    return data


def fetch_tick_data(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1m"
) -> pd.DataFrame:
    """
    Fetch tick-level data (1-minute bars for simulation).

    Args:
        symbol: Symbol to fetch
        start_date: Start date
        end_date: End date
        interval: Interval (minimum 1m)

    Returns:
        DataFrame with tick data
    """
    data = fetch_ohlcv(symbol, start_date, end_date, interval, progress=False)

    if len(data.columns.levels[0]) > 0:
        symbol_data = data[symbol].copy()
    else:
        symbol_data = data.copy()

    # Calculate additional fields
    symbol_data['dollar_volume'] = symbol_data['Close'] * symbol_data['Volume']
    symbol_data['returns'] = symbol_data['Close'].pct_change()

    return symbol_data


def prepare_training_data(
    symbols: Union[str, List[str]],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d",
    lookback_days: int = 365
) -> pd.DataFrame:
    """
    Prepare data for training with automatic date handling.

    Args:
        symbols: Symbol or list of symbols
        start_date: Start date (if None, uses lookback_days)
        end_date: End date (if None, uses today)
        interval: Data interval
        lookback_days: Days to look back if start_date is None

    Returns:
        Prepared DataFrame
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    if start_date is None:
        start = datetime.now() - timedelta(days=lookback_days)
        start_date = start.strftime('%Y-%m-%d')

    data = fetch_ohlcv(symbols, start_date, end_date, interval)

    # Clean data
    data = data.dropna()

    return data


def fetch_market_data_with_features(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch market data and add basic features.

    Args:
        symbol: Symbol to fetch
        start_date: Start date
        end_date: End date
        interval: Data interval

    Returns:
        DataFrame with OHLCV and basic features
    """
    data = fetch_ohlcv(symbol, start_date, end_date, interval)

    # Extract single symbol data
    if isinstance(data.columns, pd.MultiIndex):
        df = data[symbol].copy()
    else:
        df = data.copy()

    # Add basic features
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = pd.Series(index=df.index, dtype=float)
    df.loc[df['Close'] > 0, 'log_returns'] = pd.Series(
        pd.Series(df['Close']).apply(lambda x: x if x > 0 else None).pct_change().apply(
            lambda x: pd.Series([x]).apply(lambda y: 0 if pd.isna(y) or y <= -1 else float('inf') if y == float('inf') else __import__('numpy').log1p(y)).iloc[0]
        )
    )

    # Simpler log returns calculation
    df['log_returns'] = (df['Close'] / df['Close'].shift(1)).apply(
        lambda x: 0 if pd.isna(x) or x <= 0 else pd.Series([x]).apply(lambda y: __import__('numpy').log(y)).iloc[0]
    )

    df['dollar_volume'] = df['Close'] * df['Volume']
    df['hl_range'] = df['High'] - df['Low']
    df['oc_range'] = abs(df['Close'] - df['Open'])

    # Price momentum
    df['price_momentum_5'] = df['Close'].pct_change(5)
    df['price_momentum_20'] = df['Close'].pct_change(20)

    # Volume features
    df['volume_ma_20'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma_20']

    return df


def get_spy_data(
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to get SPY data.

    Args:
        start_date: Start date
        end_date: End date (defaults to today)

    Returns:
        SPY DataFrame
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    return fetch_market_data_with_features("SPY", start_date, end_date)


def validate_data(df: pd.DataFrame, min_rows: int = 100) -> bool:
    """
    Validate that data is suitable for training.

    Args:
        df: DataFrame to validate
        min_rows: Minimum required rows

    Returns:
        True if valid, raises ValueError otherwise
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    if len(df) < min_rows:
        raise ValueError(f"Insufficient data: {len(df)} rows, minimum {min_rows}")

    # Check for required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check for NaN values
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        print(f"Warning: NaN values found in columns: {nan_cols}")

    return True
