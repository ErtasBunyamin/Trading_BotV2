"""Randomized trading strategy."""

from __future__ import annotations

import random
from typing import List


class RandomStrategy:
    """Randomly issue buy or sell signals."""

    name = "Random"

    def __init__(
        self,
        profit_threshold: float | None = None,
        trailing_stop_pct: float | None = None,
    ) -> None:
        self.profit_threshold = profit_threshold
        self.trailing_stop_pct = trailing_stop_pct

    def generate_signals(self, prices: List[float]) -> List[tuple[int, str, float]]:
        signals: List[tuple[int, str, float]] = []
        for i in range(len(prices)):
            r = random.random()
            if r < 0.02:
                strength = random.random()
                signals.append((i, "BUY", strength))
            elif r > 0.98:
                strength = random.random()
                signals.append((i, "SELL", strength))
        return signals
