"""Moving Average Convergence Divergence strategy."""

from __future__ import annotations

from typing import List


def ema(prices: List[float], period: int) -> List[float]:
    ema_values: List[float] = []
    k = 2 / (period + 1)
    for i, price in enumerate(prices):
        if i == 0:
            ema_values.append(price)
        else:
            ema_values.append(price * k + ema_values[-1] * (1 - k))
    return ema_values


class MACDStrategy:
    """Buy when MACD crosses above signal, sell when crosses below."""

    name = "MACD"

    def __init__(
        self,
        profit_threshold: float | None = None,
        trailing_stop_pct: float | None = None,
    ) -> None:
        self.profit_threshold = profit_threshold
        self.trailing_stop_pct = trailing_stop_pct

    def generate_signals(self, prices: List[float]) -> List[tuple[int, str, float]]:
        short = ema(prices, 12)
        long_ = ema(prices, 26)
        macd_line = [s - l for s, l in zip(short, long_)]
        signal_line = ema(macd_line, 9)
        signals: List[tuple[int, str, float]] = []
        max_diff = max(abs(m - s) for m, s in zip(macd_line, signal_line)) or 1.0
        for i in range(1, len(prices)):
            if macd_line[i - 1] < signal_line[i - 1] and macd_line[i] > signal_line[i]:
                strength = abs(macd_line[i] - signal_line[i]) / max_diff
                signals.append((i, "BUY", strength))
            elif macd_line[i - 1] > signal_line[i - 1] and macd_line[i] < signal_line[i]:
                strength = abs(macd_line[i] - signal_line[i]) / max_diff
                signals.append((i, "SELL", strength))
        return signals
