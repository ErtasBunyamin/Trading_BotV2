"""MACD-based trading strategy."""

from __future__ import annotations

from collections import deque


def _ema(values: list[float], period: int) -> float:
    """Compute the Exponential Moving Average."""
    k = 2 / (period + 1)
    ema = values[0]
    for price in values[1:]:
        ema = price * k + ema * (1 - k)
    return ema


class MACDStrategy:
    """Moving Average Convergence Divergence strategy."""

    def __init__(self, short: int = 12, long: int = 26, signal: int = 9) -> None:
        self.short = short
        self.long = long
        self.signal = signal
        self.prices: deque[float] = deque(maxlen=long + signal)
        self._macd_hist: list[float] = []

    def _compute_macd(self) -> tuple[float, float]:
        prices = list(self.prices)
        ema_short = _ema(prices[-self.short :], self.short)
        ema_long = _ema(prices[-self.long :], self.long)
        macd = ema_short - ema_long
        self._macd_hist.append(macd)
        if len(self._macd_hist) < self.signal:
            signal_line = macd
        else:
            signal_line = _ema(self._macd_hist[-self.signal :], self.signal)
        return macd, signal_line

    def on_price(self, price: float) -> str:
        """Update price history and return trading signal."""
        self.prices.append(price)
        if len(self.prices) < self.long + self.signal:
            return "hold"
        macd, signal_line = self._compute_macd()
        if macd > signal_line:
            return "buy"
        if macd < signal_line:
            return "sell"
        return "hold"
