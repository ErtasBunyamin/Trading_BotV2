"""Hybrid trading strategy combining multiple indicators."""

from __future__ import annotations

from typing import List

from .bollinger import BollingerStrategy
from .macd import MACDStrategy
from .ma_cross import MACrossStrategy
from .rsi import RSIStrategy


class CustomStrategy:
    """Generate signals based on a consensus of other strategies.

    The ``threshold`` value controls how strong the combined score must be
    before emitting a signal.  A lower threshold results in more frequent
    signals.
    """

    name = "Hybrid"

    def __init__(
        self,
        profit_threshold: float | None = None,
        trailing_stop_pct: float | None = None,
        threshold: float = 0.5,
    ) -> None:
        self.profit_threshold = profit_threshold
        self.trailing_stop_pct = trailing_stop_pct
        self.threshold = threshold
        self._strategies = [
            RSIStrategy(profit_threshold, trailing_stop_pct),
            MACDStrategy(profit_threshold, trailing_stop_pct),
            BollingerStrategy(profit_threshold, trailing_stop_pct),
            MACrossStrategy(profit_threshold, trailing_stop_pct),
        ]

    def generate_signals(self, prices: List[float]) -> List[tuple[int, str, float]]:
        """Aggregate signals from all sub-strategies."""
        # Collect signals from each strategy as index -> (action, strength)
        strategy_maps = []
        for strat in self._strategies:
            mapping = {i: (a, s) for i, a, s in strat.generate_signals(prices)}
            strategy_maps.append(mapping)

        signals: List[tuple[int, str, float]] = []
        for i in range(len(prices)):
            score = 0.0
            for mapping in strategy_maps:
                action_strength = mapping.get(i)
                if action_strength is None:
                    continue
                action, strength = action_strength
                if action == "BUY":
                    score += strength
                elif action == "SELL":
                    score -= strength
            if score >= self.threshold:
                strength = min(1.0, abs(score) / len(strategy_maps))
                signals.append((i, "BUY", strength))
            elif score <= -self.threshold:
                strength = min(1.0, abs(score) / len(strategy_maps))
                signals.append((i, "SELL", strength))
        return signals
