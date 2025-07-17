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
        # Fetch 24 hours of data using 5 minute candles (288 total)
        prices = self.data_service.get_historical_prices(limit=288, interval="5m")
        # Shared account for all strategies
        balance = 10000.0
        position = 0.0

        # Precompute signals per strategy
        signals_map: dict = {}
        index_map: dict = {}
        trades_map: dict = {}
        bought_map: dict = {}
        sold_map: dict = {}
        profit_map: dict = {}
        for strategy in self.strategies:
            signals_map[strategy] = strategy.generate_signals(prices)
            index_map[strategy] = 0
            trades_map[strategy] = []
            bought_map[strategy] = 0.0
            sold_map[strategy] = 0.0
            profit_map[strategy] = 0.0

        for i, price in enumerate(prices):
            for strategy in self.strategies:
                idx = index_map[strategy]
                signals = signals_map[strategy]
                while idx < len(signals) and signals[idx][0] == i:
                    _, action, strength = signals[idx]
                    strength = max(0.0, min(1.0, strength))
                    prev_value = balance + position * price
                    if action == "BUY" and balance > 0:
                        cost = balance * strength
                        amount = cost / price
                        position += amount
                        balance -= cost
                        trades_map[strategy].append((i, "BUY", amount))
                        bought_map[strategy] += amount
                    elif action == "SELL" and position > 0:
                        amount = position * strength
                        balance += amount * price
                        position -= amount
                        trades_map[strategy].append((i, "SELL", amount))
                        sold_map[strategy] += amount
                    new_value = balance + position * price
                    profit_map[strategy] += new_value - prev_value
                    idx += 1
                index_map[strategy] = idx

        final_value = balance + position * prices[-1]
        overall_profit = final_value - 10000.0

        results = []
        for strategy in self.strategies:
            profit = profit_map[strategy]
            profit_pct = (profit / 10000.0) * 100
            results.append(
                {
                    "name": strategy.name,
                    "prices": prices,
                    "trades": trades_map[strategy],
                    "profit": profit,
                    "profit_pct": profit_pct,
                    "bought": bought_map[strategy],
                    "sold": sold_map[strategy],
                }
            )
            self.logger.log(f"{strategy.name} profit: {profit:.2f}")
        self.logger.log(f"Overall profit: {overall_profit:.2f}")
        return results
