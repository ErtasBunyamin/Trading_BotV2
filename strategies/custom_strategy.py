"""Custom trading strategy placeholder."""

from __future__ import annotations

import random


class CustomStrategy:
    """A naive custom strategy choosing random actions."""

    def on_price(self, price: float) -> str:
        """Return a random trading signal."""
        return random.choice(["buy", "sell", "hold"])
