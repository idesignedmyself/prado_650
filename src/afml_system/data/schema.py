"""
Data schema definitions for PRADO9.
Defines dataclasses for bars and market data structures.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import pandas as pd


@dataclass
class Bar:
    """Represents a single price bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bar_type: str = "time"  # time, dollar, volume, volatility

    @property
    def typical_price(self) -> float:
        """Calculate typical price (HLC/3)."""
        return (self.high + self.low + self.close) / 3

    @property
    def range(self) -> float:
        """Calculate high-low range."""
        return self.high - self.low

    @property
    def body(self) -> float:
        """Calculate candle body size."""
        return abs(self.close - self.open)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'bar_type': self.bar_type
        }


@dataclass
class BarCollection:
    """Collection of bars with utility methods."""
    bars: list[Bar] = field(default_factory=list)
    symbol: str = "UNKNOWN"

    def to_dataframe(self) -> pd.DataFrame:
        """Convert bars to DataFrame."""
        if not self.bars:
            return pd.DataFrame()

        data = [bar.to_dict() for bar in self.bars]
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

    def add_bar(self, bar: Bar):
        """Add a bar to collection."""
        self.bars.append(bar)

    def get_latest(self, n: int = 1) -> list[Bar]:
        """Get latest n bars."""
        return self.bars[-n:] if self.bars else []

    def __len__(self) -> int:
        return len(self.bars)


@dataclass
class MarketData:
    """Complete market data snapshot."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    # Optional fields
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None

    # Derived fields
    dollar_volume: Optional[float] = None
    vwap: Optional[float] = None

    def __post_init__(self):
        """Calculate derived fields."""
        if self.dollar_volume is None:
            self.dollar_volume = self.close * self.volume

        if self.vwap is None:
            self.vwap = self.close  # Simplified

    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price from bid-ask."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None


@dataclass
class OHLCV:
    """Simple OHLCV data structure."""
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'Open': self.open,
            'High': self.high,
            'Low': self.low,
            'Close': self.close,
            'Volume': self.volume
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'OHLCV':
        """Create from dictionary."""
        return cls(
            open=data.get('Open', data.get('open')),
            high=data.get('High', data.get('high')),
            low=data.get('Low', data.get('low')),
            close=data.get('Close', data.get('close')),
            volume=data.get('Volume', data.get('volume'))
        )

    @classmethod
    def from_series(cls, series: pd.Series) -> 'OHLCV':
        """Create from pandas Series."""
        return cls(
            open=series['Open'],
            high=series['High'],
            low=series['Low'],
            close=series['Close'],
            volume=series['Volume']
        )


def dataframe_to_bars(df: pd.DataFrame, bar_type: str = "time") -> list[Bar]:
    """
    Convert DataFrame to list of Bar objects.

    Args:
        df: DataFrame with OHLCV columns
        bar_type: Type of bars

    Returns:
        List of Bar objects
    """
    bars = []

    for idx, row in df.iterrows():
        bar = Bar(
            timestamp=idx,
            open=row['Open'],
            high=row['High'],
            low=row['Low'],
            close=row['Close'],
            volume=row['Volume'],
            bar_type=bar_type
        )
        bars.append(bar)

    return bars


def bars_to_dataframe(bars: list[Bar]) -> pd.DataFrame:
    """
    Convert list of Bar objects to DataFrame.

    Args:
        bars: List of Bar objects

    Returns:
        DataFrame with OHLCV columns
    """
    if not bars:
        return pd.DataFrame()

    data = []
    for bar in bars:
        data.append({
            'timestamp': bar.timestamp,
            'Open': bar.open,
            'High': bar.high,
            'Low': bar.low,
            'Close': bar.close,
            'Volume': bar.volume
        })

    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df
