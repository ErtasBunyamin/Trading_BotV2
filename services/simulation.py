from __future__ import annotations

"""Trading simulation logic for multiple strategies."""

from typing import Iterable

from services.data_service import DataService
from services.logger import Logger


class Simulation:
    """Simulate trading strategies on historical data."""

    def __init__(self, data_service: DataService, logger: Logger, strategies: Iterable) -> None:
        self.data_service = data_service
        self.logger = logger
        self.strategies = list(strategies)
        self.initial_balance = 10000.0
        self.balances = {s: self.initial_balance for s in self.strategies}
        self.positions = {s: 0.0 for s in self.strategies}
        self.signals: dict = {s: [] for s in self.strategies}
        self.prices: list[float] = []

    def run(self) -> dict:
        """Run the simulation and return results."""
        self.prices = self.data_service.fetch_historical_prices()
        for idx, price in enumerate(self.prices):
            for strat in self.strategies:
                action = strat.on_price(price)
                if action == "buy" and self.balances[strat] > 0:
                    qty = self.balances[strat] / price
                    self.positions[strat] = qty
                    self.balances[strat] = 0.0
                    self.signals[strat].append((idx, price, "buy"))
                elif action == "sell" and self.positions[strat] > 0:
                    self.balances[strat] = self.positions[strat] * price
                    self.positions[strat] = 0.0
                    self.signals[strat].append((idx, price, "sell"))
        results = {}
        last_price = self.prices[-1] if self.prices else 0.0
        for strat in self.strategies:
            final_balance = self.balances[strat] + self.positions[strat] * last_price
            profit = final_balance - self.initial_balance
            results[strat.__class__.__name__] = {
                "name": strat.__class__.__name__,
                "prices": self.prices,
                "signals": self.signals[strat],
                "profit": profit,
            }
            self.logger.log(f"{strat.__class__.__name__} profit: {profit:.2f}")
        return results
