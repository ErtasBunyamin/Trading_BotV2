"""Moving Average Cross strategy."""

from __future__ import annotations

from typing import List


class MACrossStrategy:
    """Buy when short MA crosses above long MA, sell when opposite."""

    name = "MA Cross"

    def __init__(
        self,
        profit_threshold: float | None = None,
        trailing_stop_pct: float | None = None,
    ) -> None:
        self.profit_threshold = profit_threshold
        self.trailing_stop_pct = trailing_stop_pct

    def generate_signals(self, prices: List[float]) -> List[tuple[int, str, float]]:
        short_period = 5
        long_period = 20
        signals: List[tuple[int, str, float]] = []
        for i in range(long_period, len(prices)):
            short_avg = sum(prices[i - short_period : i]) / short_period
            long_avg = sum(prices[i - long_period : i]) / long_period
            prev_short_avg = sum(prices[i - short_period - 1 : i - 1]) / short_period
            prev_long_avg = sum(prices[i - long_period - 1 : i - 1]) / long_period
            if prev_short_avg <= prev_long_avg and short_avg > long_avg:
                diff = abs(short_avg - long_avg)
                strength = min(1.0, diff / long_avg)
                signals.append((i, "BUY", strength))
            elif prev_short_avg >= prev_long_avg and short_avg < long_avg:
                diff = abs(short_avg - long_avg)
                strength = min(1.0, diff / long_avg)
                signals.append((i, "SELL", strength))
        return signals
