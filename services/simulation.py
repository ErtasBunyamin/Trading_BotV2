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

        results = []
        for strategy in self.strategies:
            signals = strategy.generate_signals(prices)
            idx = 0
            balance = 10000.0
            position = 0.0
            position_cost = 0.0
            trades: List[tuple[int, str, float, float, float]] = []
            bought_total = 0.0
            sold_total = 0.0

            for i, price in enumerate(prices):
                while idx < len(signals) and signals[idx][0] == i:
                    _, action, strength = signals[idx]
                    strength = max(0.0, min(1.0, strength))
                    if action == "BUY" and balance > 0:
                        cost = balance * strength
                        amount = cost / price
                        position += amount
                        balance -= cost
                        position_cost += cost
                        trades.append((i, "BUY", amount, price, balance))
                        bought_total += amount
                    elif action == "SELL" and position > 0:
                        # Consider selling all if profit would exceed 2% even with low strength
                        potential_profit = position * price - position_cost
                        if strength < 0.5 and position_cost > 0 and potential_profit / position_cost >= 0.02:
                            amount = position
                        else:
                            amount = position * strength
                        if amount > position:
                            amount = position
                        sell_cost = (position_cost / position) * amount
                        balance += amount * price
                        position -= amount
                        position_cost -= sell_cost
                        trades.append((i, "SELL", amount, price, balance))
                        sold_total += amount
                    idx += 1

            holding_value = position * prices[-1]
            final_value = balance + holding_value
            profit = final_value - 10000.0
            profit_pct = (profit / 10000.0) * 100

            results.append(
                {
                    "name": strategy.name,
                    "prices": prices,
                    "trades": trades,
                    "profit": profit,
                    "final_balance": final_value,
                    "profit_pct": profit_pct,
                    "bought": bought_total,
                    "sold": sold_total,
                    "remaining_btc": position,
                    "holding_value": holding_value,
                }
            )
            self.logger.log(f"{strategy.name} profit: {profit:.2f}")

        return results
