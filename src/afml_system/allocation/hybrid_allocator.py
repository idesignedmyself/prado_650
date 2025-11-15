"""
Hybrid position sizing allocator.
Combines Kelly criterion, volatility targeting, and risk parity.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional


class HybridAllocator:
    """
    Hybrid position sizing using multiple methods.

    Combines:
    1. Kelly criterion for optimal sizing
    2. Volatility targeting for risk control
    3. Risk parity for diversification
    """

    def __init__(
        self,
        max_position_size: float = 0.1,
        kelly_fraction: float = 0.25,
        volatility_target: float = 0.15,
        leverage_limit: float = 2.0
    ):
        """
        Initialize hybrid allocator.

        Args:
            max_position_size: Maximum position size (fraction of portfolio)
            kelly_fraction: Fraction of Kelly to use (for safety)
            volatility_target: Target portfolio volatility
            leverage_limit: Maximum leverage
        """
        self.max_position_size = max_position_size
        self.kelly_fraction = kelly_fraction
        self.volatility_target = volatility_target
        self.leverage_limit = leverage_limit

    def calculate_position_size(
        self,
        signal: float,
        confidence: float,
        current_volatility: float,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None
    ) -> float:
        """
        Calculate position size using hybrid method.

        Args:
            signal: Trading signal (-1 to 1)
            confidence: Prediction confidence (0 to 1)
            current_volatility: Current asset volatility
            win_rate: Historical win rate
            avg_win: Average win size
            avg_loss: Average loss size

        Returns:
            Position size (fraction of portfolio)
        """
        # Base size from signal strength
        base_size = abs(signal) * confidence

        # Kelly sizing
        if win_rate is not None and avg_win is not None and avg_loss is not None:
            kelly_size = self._kelly_criterion(win_rate, avg_win, avg_loss)
            kelly_size *= self.kelly_fraction  # Use fraction for safety
        else:
            kelly_size = base_size

        # Volatility targeting
        vol_size = self._volatility_target_size(current_volatility)

        # Combine sizes
        position_size = min(base_size, kelly_size, vol_size)

        # Apply limits
        position_size = np.clip(position_size, 0, self.max_position_size)

        # Apply direction
        position_size *= np.sign(signal)

        return position_size

    def _kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly criterion position size.

        Args:
            win_rate: Probability of winning
            avg_win: Average win size
            avg_loss: Average loss size

        Returns:
            Kelly position size
        """
        if avg_loss <= 0:
            return 0.0

        # Kelly formula: (p*W - (1-p)*L) / W
        # where p=win_rate, W=avg_win, L=avg_loss
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

        return max(0, kelly)

    def _volatility_target_size(self, current_volatility: float) -> float:
        """
        Calculate size for volatility targeting.

        Args:
            current_volatility: Current volatility

        Returns:
            Position size
        """
        if current_volatility <= 0:
            return 0.0

        size = self.volatility_target / current_volatility

        return min(size, self.leverage_limit)

    def allocate_portfolio(
        self,
        signals: Dict[str, float],
        confidences: Dict[str, float],
        volatilities: Dict[str, float],
        method: str = 'risk_parity'
    ) -> Dict[str, float]:
        """
        Allocate across multiple assets/strategies.

        Args:
            signals: Dictionary of asset -> signal
            confidences: Dictionary of asset -> confidence
            volatilities: Dictionary of asset -> volatility
            method: Allocation method ('risk_parity', 'equal_weight', 'signal_weight')

        Returns:
            Dictionary of asset -> position_size
        """
        if method == 'risk_parity':
            return self._risk_parity_allocation(signals, volatilities)
        elif method == 'equal_weight':
            return self._equal_weight_allocation(signals)
        elif method == 'signal_weight':
            return self._signal_weight_allocation(signals, confidences)
        else:
            raise ValueError(f"Unknown allocation method: {method}")

    def _risk_parity_allocation(
        self,
        signals: Dict[str, float],
        volatilities: Dict[str, float]
    ) -> Dict[str, float]:
        """Risk parity allocation."""
        allocations = {}

        # Calculate inverse volatility weights
        inv_vols = {k: 1.0 / (v + 1e-10) for k, v in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())

        # Normalize to sum to 1
        for asset, signal in signals.items():
            if asset in inv_vols:
                weight = inv_vols[asset] / total_inv_vol
                allocations[asset] = np.sign(signal) * weight
            else:
                allocations[asset] = 0.0

        return allocations

    def _equal_weight_allocation(self, signals: Dict[str, float]) -> Dict[str, float]:
        """Equal weight allocation."""
        n_assets = len(signals)
        weight = 1.0 / n_assets if n_assets > 0 else 0.0

        return {k: np.sign(v) * weight for k, v in signals.items()}

    def _signal_weight_allocation(
        self,
        signals: Dict[str, float],
        confidences: Dict[str, float]
    ) -> Dict[str, float]:
        """Signal strength weighted allocation."""
        allocations = {}

        # Calculate weights from signal strength and confidence
        weights = {k: abs(v) * confidences.get(k, 1.0) for k, v in signals.items()}
        total_weight = sum(weights.values())

        if total_weight > 0:
            for asset, signal in signals.items():
                allocations[asset] = (
                    np.sign(signal) * weights[asset] / total_weight
                )
        else:
            allocations = {k: 0.0 for k in signals.keys()}

        return allocations

    def apply_risk_limits(
        self,
        allocations: Dict[str, float],
        current_positions: Optional[Dict[str, float]] = None,
        max_turnover: float = 0.5
    ) -> Dict[str, float]:
        """
        Apply risk limits to allocations.

        Args:
            allocations: Target allocations
            current_positions: Current positions
            max_turnover: Maximum portfolio turnover

        Returns:
            Risk-limited allocations
        """
        limited = allocations.copy()

        # Limit individual positions
        for asset in limited:
            limited[asset] = np.clip(
                limited[asset],
                -self.max_position_size,
                self.max_position_size
            )

        # Limit total leverage
        total_leverage = sum(abs(v) for v in limited.values())
        if total_leverage > self.leverage_limit:
            scale = self.leverage_limit / total_leverage
            limited = {k: v * scale for k, v in limited.items()}

        # Limit turnover if current positions provided
        if current_positions is not None:
            turnover = sum(
                abs(limited.get(k, 0) - current_positions.get(k, 0))
                for k in set(limited.keys()) | set(current_positions.keys())
            )

            if turnover > max_turnover:
                # Scale down changes
                scale = max_turnover / turnover
                for asset in limited:
                    current = current_positions.get(asset, 0)
                    change = limited[asset] - current
                    limited[asset] = current + change * scale

        return limited
