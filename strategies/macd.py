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

    def generate_signals(self, prices: List[float]) -> List[tuple[int, str]]:
        short = ema(prices, 12)
        long_ = ema(prices, 26)
        macd_line = [s - l for s, l in zip(short, long_)]
        signal_line = ema(macd_line, 9)
        signals: List[tuple[int, str]] = []
        for i in range(1, len(prices)):
            if macd_line[i - 1] < signal_line[i - 1] and macd_line[i] > signal_line[i]:
                signals.append((i, "BUY"))
            elif macd_line[i - 1] > signal_line[i - 1] and macd_line[i] < signal_line[i]:
                signals.append((i, "SELL"))
        return signals
