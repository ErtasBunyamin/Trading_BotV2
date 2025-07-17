"""Trading simulation logic using supplied strategies."""

from __future__ import annotations

from typing import Iterable, List
import time

from services.data_service import DataService
from services.logger import Logger


class Simulation:
    """Simulate trades for each strategy using live price data."""

    def __init__(self, data_service: DataService, logger: Logger, strategies: Iterable) -> None:
        self.data_service = data_service
        self.logger = logger
        self.strategies = list(strategies)

    def run(self, iterations: int = 100, interval: float = 300.0) -> List[dict]:
        """Run the simulation in real-time and return results per strategy."""
        results = []
        for strategy in self.strategies:
            results.append({
                "name": strategy.name,
                "prices": [],
                "trades": [],
                "balance": 10000.0,
                "position": 0.0,
            })

        prices: List[float] = []
        for i in range(iterations):
            price = self.data_service.fetch_price()
            self.logger.log(
                f"Price update {i + 1}/{iterations}: {price:.2f}"
            )
            prices.append(price)
            for result, strategy in zip(results, self.strategies):
                result["prices"].append(price)
                signals = strategy.generate_signals(prices)
                last_index = len(prices) - 1
                for j, action, strength in signals:
                    if j != last_index:
                        continue
                    strength = max(0.0, min(1.0, strength))
                    if action == "BUY" and result["balance"] > 0:
                        amount = result["balance"] * strength
                        result["position"] += amount / price
                        result["balance"] -= amount
                        result["trades"].append((i, "BUY"))
                        self.logger.log(
                            f"{strategy.name} BUY {amount:.2f} at {price:.2f}"
                        )
                    elif action == "SELL" and result["position"] > 0:
                        amount = result["position"] * strength
                        result["balance"] += amount * price
                        result["position"] -= amount
                        result["trades"].append((i, "SELL"))
                        self.logger.log(
                            f"{strategy.name} SELL {amount:.2f} at {price:.2f}"
                        )
            if i < iterations - 1:
                self.logger.log(f"Waiting {interval} seconds for next update")
                time.sleep(interval)

        for result in results:
            final_balance = result["balance"] + result["position"] * prices[-1]
            result["profit"] = final_balance - 10000.0
            self.logger.log(f"{result['name']} profit: {result['profit']:.2f}")

        # remove helper fields
        for result in results:
            result.pop("balance")
            result.pop("position")
        return results
