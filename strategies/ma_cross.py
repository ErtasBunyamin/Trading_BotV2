"""Moving Average Cross strategy."""

from __future__ import annotations

from typing import List


class MACrossStrategy:
    """Buy when short MA crosses above long MA, sell when opposite."""

    name = "MA Cross"

    def generate_signals(self, prices: List[float]) -> List[tuple[int, str]]:
        short_period = 5
        long_period = 20
        signals: List[tuple[int, str]] = []
        for i in range(long_period, len(prices)):
            short_avg = sum(prices[i - short_period : i]) / short_period
            long_avg = sum(prices[i - long_period : i]) / long_period
            prev_short_avg = sum(prices[i - short_period - 1 : i - 1]) / short_period
            prev_long_avg = sum(prices[i - long_period - 1 : i - 1]) / long_period
            if prev_short_avg <= prev_long_avg and short_avg > long_avg:
                signals.append((i, "BUY"))
            elif prev_short_avg >= prev_long_avg and short_avg < long_avg:
                signals.append((i, "SELL"))
        return signals
