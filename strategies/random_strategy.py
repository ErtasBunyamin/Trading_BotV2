"""Randomized trading strategy."""

from __future__ import annotations

import random
from typing import List


class RandomStrategy:
    """Randomly issue buy or sell signals."""

    name = "Random"

    def generate_signals(self, prices: List[float]) -> List[tuple[int, str]]:
        signals: List[tuple[int, str]] = []
        for i in range(len(prices)):
            r = random.random()
            if r < 0.02:
                signals.append((i, "BUY"))
            elif r > 0.98:
                signals.append((i, "SELL"))
        return signals
