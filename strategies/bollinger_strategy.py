"""Bollinger Bands based trading strategy."""

from __future__ import annotations

from collections import deque
from statistics import mean, stdev


class BollingerStrategy:
    """Trading signals generated using Bollinger Bands."""

    def __init__(self, period: int = 20, dev_factor: float = 2.0) -> None:
        self.period = period
        self.dev_factor = dev_factor
        self.prices: deque[float] = deque(maxlen=period)

    def _bands(self) -> tuple[float, float]:
        prices = list(self.prices)
        mid = mean(prices)
        if len(prices) < 2:
            return mid, mid
        deviation = stdev(prices)
        upper = mid + self.dev_factor * deviation
        lower = mid - self.dev_factor * deviation
        return upper, lower

    def on_price(self, price: float) -> str:
        """Update price history and return trading signal."""
        self.prices.append(price)
        if len(self.prices) < self.period:
            return "hold"
        upper, lower = self._bands()
        if price < lower:
            return "buy"
        if price > upper:
            return "sell"
        return "hold"
