"""Dynamic hybrid trading strategy using multiple indicators.

This strategy aggregates signals from several indicators and adjusts
its decision threshold based on market regime and recent performance of
each indicator. It also factors in the current price location within a
lookback window and the short-term trend direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from .rsi import RSIStrategy
from .macd import MACDStrategy
from .bollinger import BollingerStrategy
from .ma_cross import MACrossStrategy


class DynamicHybridStrategy:
    """Generate signals with adaptive thresholds based on market regime."""

    name = "Dynamic Hybrid"

    def __init__(
        self,
        base_threshold: float = 0.18,
        volatile_threshold: float = 0.25,
        sideways_threshold: float = 0.11,
        lookback: int = 50,
        *,
        std_coef: float = 0.004,
        slope_threshold: float = 0.0002,
        location_weight: float = 0.2,
        trend_weight: float = 0.2,
        min_signal_distance: int = 1,
        confidence_threshold: float = 0.0,
    ) -> None:
        self.indicator_winrates: Dict[str, float] = {
            "RSI": 0.5,
            "MACD": 0.5,
            "Bollinger": 0.5,
            "MA Cross": 0.5,
        }
        self.market_regime = "trend"
        self.base_threshold = base_threshold
        self.volatile_threshold = volatile_threshold
        self.sideways_threshold = sideways_threshold
        self.lookback = lookback
        self.std_coef = std_coef
        self.slope_threshold = slope_threshold
        self.location_weight = location_weight
        self.trend_weight = trend_weight
        self.min_signal_distance = min_signal_distance
        self.confidence_threshold = confidence_threshold

        # Sub-strategies providing raw signals
        self._strategies = [
            RSIStrategy(),
            MACDStrategy(),
            BollingerStrategy(),
            MACrossStrategy(),
        ]

    def update_winrates(self, trade_logs: List[Dict]) -> None:
        """Update indicator win rates using the last 20 trades."""
        for ind in self.indicator_winrates:
            trades = [t for t in trade_logs if t["indicator"] == ind][-20:]
            if trades:
                win = sum(1 for t in trades if t["pnl"] > 0)
                self.indicator_winrates[ind] = win / len(trades)
            else:
                self.indicator_winrates[ind] = 0.5

    def detect_market_regime(self, prices: List[float]) -> None:
        """Classify the market regime using EMA slope and volatility."""
        ema = pd.Series(prices).ewm(span=self.lookback).mean()
        window = prices[-self.lookback :]
        std = np.std(window)
        if len(ema) >= self.lookback:
            start_idx = -self.lookback
            length = self.lookback
        else:
            start_idx = 0
            length = max(1, len(ema) - 1)
        slope = (ema.iloc[-1] - ema.iloc[start_idx]) / length
        mean_price = np.mean(window)
        if std > self.std_coef * mean_price:
            self.market_regime = "volatile"
        elif abs(slope) < self.slope_threshold:
            self.market_regime = "sideways"
        else:
            self.market_regime = "trend"

    def dynamic_voting(self, indicator_signals: Dict[str, Tuple[str, float]]) -> float:
        """Vote on the final score using indicator win rates."""
        score = 0.0
        total_weight = sum(self.indicator_winrates.values())
        for ind, (action, strength) in indicator_signals.items():
            if ind not in self.indicator_winrates:
                # Register unseen indicator with a neutral win rate
                self.indicator_winrates[ind] = 0.5
                total_weight = sum(self.indicator_winrates.values())
            weight = (
                self.indicator_winrates[ind] / total_weight if total_weight else 0.25
            )
            if action == "BUY":
                score += strength * weight
            elif action == "SELL":
                score -= strength * weight
        return score

    def get_threshold(self) -> float:
        """Return the buy/sell threshold for the current market regime."""
        if self.market_regime == "trend":
            return self.base_threshold
        if self.market_regime == "volatile":
            return self.volatile_threshold
        return self.sideways_threshold

    def _trend_score(self, window: List[float]) -> float:
        """Return normalized trend strength for the given price window."""
        n = len(window)
        if n < 2:
            return 0.0
        xs = list(range(n))
        mean_x = sum(xs) / n
        mean_y = sum(window) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, window))
        den = sum((x - mean_x) ** 2 for x in xs)
        slope = num / den if den != 0 else 0.0
        trend = slope * n / window[-1]
        return max(-1.0, min(1.0, trend))

    def generate_signals(
        self,
        prices: List[float],
        strat_signals: List[Tuple[int, str, float, str]] | None = None,
        trade_logs: List[Dict] | None = None,
    ) -> List[Tuple[int, str, float]]:
        """Aggregate indicator signals and return trading decisions."""

        # When no indicator signals are supplied, gather them from the default
        # sub-strategies so this method is compatible with ``Simulation`` which
        # only passes ``prices``.
        if strat_signals is None:
            strat_signals = []
            for strat in self._strategies:
                for idx, action, strength in strat.generate_signals(prices):
                    strat_signals.append((idx, action, strength, strat.name))

        # Derive simple trade logs when not provided so indicator win rates can
        # still be updated. The outcome of each signal is evaluated on the next
        # bar only as a lightweight heuristic.
        if trade_logs is None:
            trade_logs = []
            signals_by_indicator: Dict[str, List[Tuple[int, str]]] = {}
            for idx, action, _strength, ind in strat_signals:
                signals_by_indicator.setdefault(ind, []).append((idx, action))
            for ind, sigs in signals_by_indicator.items():
                for idx, action in sigs[-20:]:
                    if idx + 1 < len(prices):
                        diff = prices[idx + 1] - prices[idx]
                        pnl = diff if action == "BUY" else -diff
                        trade_logs.append({"indicator": ind, "pnl": pnl})

        self.update_winrates(trade_logs)

        # Organize signals per bar as indicator -> (action, strength)
        signals_by_idx: Dict[int, Dict[str, Tuple[str, float]]] = {}
        for idx, action, strength, ind in strat_signals:
            signals_by_idx.setdefault(idx, {})[ind] = (action, strength)

        final_signals: List[Tuple[int, str, float]] = []
        last_idx = -self.min_signal_distance

        for idx in range(len(prices)):
            indicator_map = signals_by_idx.get(idx)
            if not indicator_map:
                continue

            self.detect_market_regime(prices[: idx + 1])
            score = self.dynamic_voting(indicator_map)

            # Price location contribution within lookback window
            start = max(0, idx - self.lookback + 1)
            window = prices[start : idx + 1]
            low = min(window)
            high = max(window)
            if high != low:
                pos = (prices[idx] - low) / (high - low)
                score += (0.5 - pos) * 2 * self.location_weight

            # Trend contribution
            if self.trend_weight:
                score += self._trend_score(window) * self.trend_weight

            threshold = self.get_threshold()

            if abs(score) < self.confidence_threshold:
                continue
            if idx - last_idx < self.min_signal_distance:
                continue

            if score >= threshold:
                final_signals.append((idx, "BUY", min(1.0, abs(score))))
                last_idx = idx
            elif score <= -threshold:
                final_signals.append((idx, "SELL", min(1.0, abs(score))))
                last_idx = idx

        return final_signals


# Example usage
# strategy = DynamicHybridStrategy()
# signals = strategy.generate_signals(prices)
