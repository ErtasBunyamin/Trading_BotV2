"""Moving Average Crossover trading strategy."""

from __future__ import annotations

from collections import deque
from statistics import mean


class MACrossStrategy:
    """Simple moving average cross strategy."""

    def __init__(self, short: int = 5, long: int = 20) -> None:
        if short >= long:
            raise ValueError("short period must be less than long period")
        self.short = short
        self.long = long
        self.short_prices: deque[float] = deque(maxlen=short)
        self.long_prices: deque[float] = deque(maxlen=long)
        self.prev_short_ma: float | None = None
        self.prev_long_ma: float | None = None

    def on_price(self, price: float) -> str:
        """Update price history and return trading signal."""
        self.short_prices.append(price)
        self.long_prices.append(price)
        if len(self.long_prices) < self.long:
            return "hold"
        short_ma = mean(self.short_prices)
        long_ma = mean(self.long_prices)
        signal = "hold"
        if self.prev_short_ma is not None and self.prev_long_ma is not None:
            if self.prev_short_ma <= self.prev_long_ma and short_ma > long_ma:
                signal = "buy"
            elif self.prev_short_ma >= self.prev_long_ma and short_ma < long_ma:
                signal = "sell"
        self.prev_short_ma = short_ma
        self.prev_long_ma = long_ma
        return signal
