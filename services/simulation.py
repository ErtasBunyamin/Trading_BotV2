"""Trading simulation logic using supplied strategies."""

from __future__ import annotations

from typing import Iterable, List

from services.data_service import DataService
from services.logger import Logger


class Simulation:
    """Simulate trades for each strategy on historical price data."""

    def __init__(self, data_service: DataService, logger: Logger, strategies: Iterable) -> None:
        self.data_service = data_service
        self.logger = logger
        self.strategies = list(strategies)

    def run(self) -> List[dict]:
        """Run the simulation and return results per strategy."""
        prices = self.data_service.get_historical_prices()
        results = []
        for strategy in self.strategies:
            balance = 10000.0
            position = 0.0
            trades: List[tuple[int, str]] = []
            signals = strategy.generate_signals(prices)
            signals_index = 0
            for i, price in enumerate(prices):
                while signals_index < len(signals) and signals[signals_index][0] == i:
                    _, action, strength = signals[signals_index]
                    strength = max(0.0, min(1.0, strength))
                    if action == "BUY" and balance > 0:
                        amount = balance * strength
                        position += amount / price
                        balance -= amount
                        trades.append((i, "BUY"))
                    elif action == "SELL" and position > 0:
                        amount = position * strength
                        balance += amount * price
                        position -= amount
                        trades.append((i, "SELL"))
                    signals_index += 1
            final_balance = balance + position * prices[-1]
            profit = final_balance - 10000.0
            results.append({
                "name": strategy.name,
                "prices": prices,
                "trades": trades,
                "profit": profit,
            })
            self.logger.log(f"{strategy.name} profit: {profit:.2f}")
        return results
