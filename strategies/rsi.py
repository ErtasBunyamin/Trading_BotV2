"""Relative Strength Index trading strategy."""

from __future__ import annotations

from typing import List


class RSIStrategy:
    """Buy when RSI < 30, sell when RSI > 70."""

    name = "RSI"

    def __init__(
        self,
        profit_threshold: float | None = None,
        trailing_stop_pct: float | None = None,
    ) -> None:
        self.profit_threshold = profit_threshold
        self.trailing_stop_pct = trailing_stop_pct

    def generate_signals(self, prices: List[float]) -> List[tuple[int, str, float]]:
        signals: List[tuple[int, str, float]] = []
        gains: List[float] = []
        losses: List[float] = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
            if i < 14:
                continue
            avg_gain = sum(gains[-14:]) / 14
            avg_loss = sum(losses[-14:]) / 14
            rs = avg_gain / avg_loss if avg_loss != 0 else 0
            rsi = 100 - 100 / (1 + rs)
            if rsi < 30:
                strength = min(1.0, (30 - rsi) / 30)
                signals.append((i, "BUY", strength))
            elif rsi > 70:
                strength = min(1.0, (rsi - 70) / 30)
                signals.append((i, "SELL", strength))
        return signals
