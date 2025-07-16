"""RSI-based trading strategy."""

from __future__ import annotations

from collections import deque


class RSIStrategy:
    """Simple Relative Strength Index strategy."""

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self.prices: deque[float] = deque(maxlen=period + 1)

    def _compute_rsi(self) -> float:
        gains = []
        losses = []
        for i in range(1, len(self.prices)):
            delta = self.prices[i] - self.prices[i - 1]
            if delta >= 0:
                gains.append(delta)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(-delta)
        avg_gain = sum(gains) / self.period
        avg_loss = sum(losses) / self.period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def on_price(self, price: float) -> str:
        """Update price history and return trading signal."""
        self.prices.append(price)
        if len(self.prices) <= self.period:
            return "hold"
        rsi = self._compute_rsi()
        if rsi < 30:
            return "buy"
        if rsi > 70:
            return "sell"
        return "hold"
