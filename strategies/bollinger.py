"""Bollinger Bands trading strategy."""

from __future__ import annotations

from typing import List
import statistics


class BollingerStrategy:
    """Buy below lower band, sell above upper band."""

    name = "Bollinger"

    def generate_signals(self, prices: List[float]) -> List[tuple[int, str]]:
        signals: List[tuple[int, str]] = []
        period = 20
        for i in range(period, len(prices)):
            window = prices[i - period : i]
            avg = sum(window) / period
            std = statistics.stdev(window)
            upper = avg + 2 * std
            lower = avg - 2 * std
            if prices[i] < lower:
                signals.append((i, "BUY"))
            elif prices[i] > upper:
                signals.append((i, "SELL"))
        return signals
