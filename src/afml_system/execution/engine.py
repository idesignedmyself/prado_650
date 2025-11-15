"""
Execution engine for trade execution.
Simulates order execution with slippage and transaction costs.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime


@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    commission: float = 0.0
    slippage: float = 0.0

    @property
    def total_cost(self) -> float:
        """Calculate total cost including fees."""
        base_cost = self.quantity * self.price
        return base_cost + self.commission + self.slippage

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'price': self.price,
            'commission': self.commission,
            'slippage': self.slippage,
            'total_cost': self.total_cost
        }


class ExecutionEngine:
    """
    Execution engine for simulating trade execution.
    """

    def __init__(
        self,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        min_commission: float = 1.0
    ):
        """
        Initialize execution engine.

        Args:
            commission_rate: Commission as fraction of trade value
            slippage_rate: Slippage as fraction of price
            min_commission: Minimum commission per trade
        """
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.min_commission = min_commission
        self.trades: List[Trade] = []

    def execute_trade(
        self,
        timestamp: datetime,
        symbol: str,
        signal: float,
        price: float,
        portfolio_value: float,
        position_size: float
    ) -> Optional[Trade]:
        """
        Execute a trade based on signal.

        Args:
            timestamp: Trade timestamp
            symbol: Symbol to trade
            signal: Trading signal (-1 to 1)
            price: Current price
            portfolio_value: Current portfolio value
            position_size: Position size as fraction of portfolio

        Returns:
            Trade object or None
        """
        if abs(signal) < 0.01:  # Minimum signal threshold
            return None

        # Calculate quantity
        dollar_amount = portfolio_value * abs(position_size)
        quantity = dollar_amount / price

        # Determine side
        side = 'buy' if signal > 0 else 'sell'

        # Calculate costs
        trade_value = quantity * price
        commission = max(trade_value * self.commission_rate, self.min_commission)

        # Slippage (worse price due to market impact)
        slippage_amount = quantity * price * self.slippage_rate
        if side == 'buy':
            execution_price = price * (1 + self.slippage_rate)
        else:
            execution_price = price * (1 - self.slippage_rate)

        # Create trade
        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=execution_price,
            commission=commission,
            slippage=slippage_amount
        )

        self.trades.append(trade)

        return trade

    def execute_rebalance(
        self,
        timestamp: datetime,
        current_positions: Dict[str, float],
        target_positions: Dict[str, float],
        prices: Dict[str, float],
        portfolio_value: float
    ) -> List[Trade]:
        """
        Execute portfolio rebalancing.

        Args:
            timestamp: Rebalance timestamp
            current_positions: Current position sizes
            target_positions: Target position sizes
            prices: Current prices
            portfolio_value: Portfolio value

        Returns:
            List of executed trades
        """
        trades = []

        all_symbols = set(current_positions.keys()) | set(target_positions.keys())

        for symbol in all_symbols:
            current = current_positions.get(symbol, 0.0)
            target = target_positions.get(symbol, 0.0)

            change = target - current

            if abs(change) > 0.01:  # Minimum change threshold
                price = prices.get(symbol, 0.0)
                if price > 0:
                    signal = np.sign(change)
                    trade = self.execute_trade(
                        timestamp, symbol, signal, price,
                        portfolio_value, abs(change)
                    )
                    if trade:
                        trades.append(trade)

        return trades

    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame([t.to_dict() for t in self.trades])

    def get_execution_stats(self) -> Dict:
        """Get execution statistics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'total_commission': 0.0,
                'total_slippage': 0.0,
                'avg_commission': 0.0,
                'avg_slippage': 0.0
            }

        df = self.get_trade_history()

        return {
            'total_trades': len(self.trades),
            'total_commission': df['commission'].sum(),
            'total_slippage': df['slippage'].sum(),
            'avg_commission': df['commission'].mean(),
            'avg_slippage': df['slippage'].mean(),
            'total_cost': df['commission'].sum() + df['slippage'].sum()
        }

    def reset(self):
        """Reset trade history."""
        self.trades = []
