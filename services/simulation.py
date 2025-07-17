"""Trading simulation logic using supplied strategies."""

from __future__ import annotations

from typing import Iterable, List

from services.data_service import DataService
from services.logger import Logger


class Simulation:
    """Simulate trades for each strategy on historical price data."""

    def __init__(
        self,
        data_service: DataService,
        logger: Logger,
        strategies: Iterable,
        trailing_stop_pct: float = 0.01,
        profit_threshold: float = 0.02,
        price_limit: int | None = 288,
        full_balance: bool = False,
    ) -> None:
        self.data_service = data_service
        self.logger = logger
        self.strategies = list(strategies)
        self.trailing_stop_pct = trailing_stop_pct
        self.profit_threshold = profit_threshold
        self.price_limit = price_limit
        self.full_balance = full_balance

    def run(self) -> List[dict]:
        """Run the simulation and return results per strategy."""
        # Fetch historical prices. ``price_limit`` may be ``None`` to request
        # the maximum number of candles from the data service.
        prices = self.data_service.get_historical_prices(
            limit=self.price_limit, interval="5m"
        )

        results = []
        for strategy in self.strategies:
            # Allow strategies to prepare using the full price history
            if hasattr(strategy, "before_run"):
                strategy.before_run(prices)
            strategy_trailing_stop = (
                getattr(strategy, "trailing_stop_pct", None)
                if getattr(strategy, "trailing_stop_pct", None) is not None
                else self.trailing_stop_pct
            )
            strategy_profit_threshold = (
                getattr(strategy, "profit_threshold", None)
                if getattr(strategy, "profit_threshold", None) is not None
                else self.profit_threshold
            )
            trade_full_balance = getattr(strategy, "trade_full_balance", self.full_balance)

            signals = strategy.generate_signals(prices)
            idx = 0
            balance = 10000.0
            position = 0.0
            position_cost = 0.0
            trades: List[tuple[int, str, float, float, float]] = []
            bought_total = 0.0
            sold_total = 0.0
            highest_price = 0.0
            trailing_closed = 0

            for i, price in enumerate(prices):
                if position > 0:
                    if price > highest_price:
                        highest_price = price
                    elif price <= highest_price * (1 - strategy_trailing_stop):
                        balance += position * price
                        trades.append((i, "SELL", position, price, balance))
                        sold_total += position
                        position = 0.0
                        position_cost = 0.0
                        highest_price = 0.0
                        trailing_closed += 1

                while idx < len(signals) and signals[idx][0] == i:
                    _, action, strength = signals[idx]
                    strength = max(0.0, min(1.0, strength))

                    if trade_full_balance:
                        strength = 1.0

                    if action == "BUY" and balance > 0:
                        cost = balance * strength
                        amount = cost / price
                        position += amount
                        balance -= cost
                        position_cost += cost
                        trades.append((i, "BUY", amount, price, balance))
                        bought_total += amount
                        if price > highest_price:
                            highest_price = price
                    elif action == "SELL" and position > 0:
                        potential_profit = position * price - position_cost
                        if trade_full_balance:
                            amount = position
                        elif (
                            strength < 0.5
                            and position_cost > 0
                            and potential_profit / position_cost >= strategy_profit_threshold
                        ):
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
                        if position == 0:
                            highest_price = 0.0
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
                    "trailing_stops": trailing_closed,
                    "profit_threshold": strategy_profit_threshold,
                    "trailing_stop_pct": strategy_trailing_stop,
                }
            )
            self.logger.log(f"{strategy.name} profit: {profit:.2f}")

        return results
