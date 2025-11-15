"""
Risk manager for position and portfolio risk control.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, List


class RiskManager:
    """
    Risk management system.

    Monitors and controls:
    - Position sizes
    - Portfolio leverage
    - Drawdowns
    - Concentration
    """

    def __init__(
        self,
        max_position_size: float = 0.1,
        max_leverage: float = 2.0,
        max_drawdown: float = 0.2,
        max_concentration: float = 0.3,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10
    ):
        """
        Initialize risk manager.

        Args:
            max_position_size: Maximum position size (fraction)
            max_leverage: Maximum leverage
            max_drawdown: Maximum drawdown before shutdown
            max_concentration: Maximum sector/strategy concentration
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.max_drawdown = max_drawdown
        self.max_concentration = max_concentration
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self.peak_value = 0.0
        self.current_drawdown = 0.0
        self.risk_limit_breached = False

    def check_position_size(
        self,
        position_size: float,
        symbol: str = None
    ) -> bool:
        """
        Check if position size is within limits.

        Args:
            position_size: Proposed position size
            symbol: Symbol name

        Returns:
            True if acceptable
        """
        if abs(position_size) > self.max_position_size:
            print(f"Position size {position_size:.2%} exceeds limit {self.max_position_size:.2%}")
            return False

        return True

    def check_portfolio_leverage(
        self,
        positions: Dict[str, float]
    ) -> bool:
        """
        Check if portfolio leverage is within limits.

        Args:
            positions: Dictionary of symbol -> position_size

        Returns:
            True if acceptable
        """
        total_leverage = sum(abs(size) for size in positions.values())

        if total_leverage > self.max_leverage:
            print(f"Portfolio leverage {total_leverage:.2f} exceeds limit {self.max_leverage:.2f}")
            return False

        return True

    def check_drawdown(
        self,
        current_value: float
    ) -> bool:
        """
        Check drawdown and update peak.

        Args:
            current_value: Current portfolio value

        Returns:
            True if acceptable
        """
        # Update peak
        if current_value > self.peak_value:
            self.peak_value = current_value

        # Calculate drawdown
        if self.peak_value > 0:
            self.current_drawdown = (self.peak_value - current_value) / self.peak_value
        else:
            self.current_drawdown = 0.0

        if self.current_drawdown > self.max_drawdown:
            print(f"Drawdown {self.current_drawdown:.2%} exceeds limit {self.max_drawdown:.2%}")
            self.risk_limit_breached = True
            return False

        return True

    def check_concentration(
        self,
        positions: Dict[str, float],
        groups: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Check concentration risk.

        Args:
            positions: Position sizes
            groups: Optional grouping (e.g., sector, strategy)

        Returns:
            True if acceptable
        """
        if groups is None:
            # No grouping - check largest single position
            max_position = max(abs(size) for size in positions.values()) if positions else 0
            if max_position > self.max_concentration:
                print(f"Single position {max_position:.2%} exceeds concentration limit")
                return False
        else:
            # Check group concentrations
            group_exposure = {}
            for symbol, size in positions.items():
                group = groups.get(symbol, 'unknown')
                group_exposure[group] = group_exposure.get(group, 0) + abs(size)

            max_group = max(group_exposure.values()) if group_exposure else 0
            if max_group > self.max_concentration:
                print(f"Group concentration {max_group:.2%} exceeds limit")
                return False

        return True

    def apply_stop_loss_take_profit(
        self,
        entry_price: float,
        current_price: float,
        side: str
    ) -> Optional[str]:
        """
        Check if stop loss or take profit triggered.

        Args:
            entry_price: Entry price
            current_price: Current price
            side: Position side ('long' or 'short')

        Returns:
            'stop_loss', 'take_profit', or None
        """
        if side == 'long':
            pnl_pct = (current_price - entry_price) / entry_price

            if pnl_pct <= -self.stop_loss_pct:
                return 'stop_loss'
            elif pnl_pct >= self.take_profit_pct:
                return 'take_profit'

        elif side == 'short':
            pnl_pct = (entry_price - current_price) / entry_price

            if pnl_pct <= -self.stop_loss_pct:
                return 'stop_loss'
            elif pnl_pct >= self.take_profit_pct:
                return 'take_profit'

        return None

    def validate_trade(
        self,
        position_size: float,
        current_positions: Dict[str, float],
        portfolio_value: float,
        symbol: str = None
    ) -> bool:
        """
        Validate if trade should be executed.

        Args:
            position_size: Proposed position size
            current_positions: Current positions
            portfolio_value: Portfolio value
            symbol: Symbol

        Returns:
            True if trade is acceptable
        """
        # Check if risk limits already breached
        if self.risk_limit_breached:
            print("Risk limits breached - no new trades allowed")
            return False

        # Check individual position size
        if not self.check_position_size(position_size, symbol):
            return False

        # Check portfolio leverage
        new_positions = current_positions.copy()
        if symbol:
            new_positions[symbol] = new_positions.get(symbol, 0) + position_size

        if not self.check_portfolio_leverage(new_positions):
            return False

        # Check drawdown
        if not self.check_drawdown(portfolio_value):
            return False

        return True

    def get_risk_metrics(
        self,
        positions: Dict[str, float],
        portfolio_value: float
    ) -> Dict:
        """
        Calculate current risk metrics.

        Args:
            positions: Current positions
            portfolio_value: Portfolio value

        Returns:
            Dictionary of risk metrics
        """
        total_leverage = sum(abs(size) for size in positions.values())
        max_position = max(abs(size) for size in positions.values()) if positions else 0

        return {
            'total_leverage': total_leverage,
            'max_position_size': max_position,
            'current_drawdown': self.current_drawdown,
            'peak_value': self.peak_value,
            'num_positions': len(positions),
            'risk_limit_breached': self.risk_limit_breached,
            'leverage_utilization': total_leverage / self.max_leverage,
            'drawdown_utilization': self.current_drawdown / self.max_drawdown
        }

    def reset(self):
        """Reset risk manager state."""
        self.peak_value = 0.0
        self.current_drawdown = 0.0
        self.risk_limit_breached = False
