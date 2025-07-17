"""Dynamic hybrid trading strategy using multiple indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class DynamicHybridStrategy:
    """Generate signals with adaptive thresholds based on market regime."""

    name = "Dynamic Hybrid"

    def __init__(
        self,
        base_threshold: float = 0.18,
        volatile_threshold: float = 0.25,
        sideways_threshold: float = 0.11,
        lookback: int = 50,
    ) -> None:
        self.indicator_winrates = {
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
        """Classify market regime as trend, volatile or sideways."""
        ema = pd.Series(prices).ewm(span=self.lookback).mean()
        std = np.std(prices[-self.lookback :])
        slope = (ema.iloc[-1] - ema.iloc[-self.lookback]) / self.lookback
        mean_price = np.mean(prices[-self.lookback :])
        if std > 0.004 * mean_price:
            self.market_regime = "volatile"
        elif abs(slope) < 0.0002:
            self.market_regime = "sideways"
        else:
            self.market_regime = "trend"

    def dynamic_voting(self, indicator_signals: Dict[str, Tuple[str, float]]) -> float:
        """Vote on the final score using indicator win rates."""
        score = 0.0
        total_weight = sum(self.indicator_winrates.values())
        for ind, (action, strength) in indicator_signals.items():
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

    def generate_signal(
        self,
        prices: List[float],
        indicator_signals: Dict[str, Tuple[str, float]],
        trade_logs: List[Dict],
    ) -> Tuple[str | None, float]:
        """Return a single action and its score."""
        self.update_winrates(trade_logs)
        self.detect_market_regime(prices)
        threshold = self.get_threshold()
        score = self.dynamic_voting(indicator_signals)
        if score >= threshold:
            return "BUY", score
        if score <= -threshold:
            return "SELL", abs(score)
        return None, 0.0
