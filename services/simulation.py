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
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        min_trade_size: float = 0.0,
    ) -> None:
        self.data_service = data_service
        self.logger = logger
        self.strategies = list(strategies)
        self.trailing_stop_pct = trailing_stop_pct
        self.profit_threshold = profit_threshold
        self.price_limit = price_limit
        self.full_balance = full_balance
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.min_trade_size = min_trade_size

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
            strategy_trailing_stop = getattr(
                strategy, "trailing_stop_pct", self.trailing_stop_pct
            )
            strategy_profit_threshold = getattr(
                strategy, "profit_threshold", self.profit_threshold
            )
            strategy_commission_pct = getattr(
                strategy, "commission_pct", self.commission_pct
            )
            strategy_slippage_pct = getattr(
                strategy, "slippage_pct", self.slippage_pct
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

            performance_logs: List[dict] = []
            trade_amounts: List[float] = []

            for i, price in enumerate(prices):
                if position > 0:
                    if price > highest_price:
                        highest_price = price
                    strategy_trailing_stop = getattr(
                        strategy, "trailing_stop_pct", self.trailing_stop_pct
                    )
                    # Skip trailing stop check when ``strategy_trailing_stop`` is
                    # ``None`` as some strategies may disable it by default.
                    if (
                        strategy_trailing_stop is not None
                        and price <= highest_price * (1 - strategy_trailing_stop)
                    ):
                        balance += position * price
                        trades.append((i, "SELL", position, price, balance))
                        performance_logs.append(
                            {
                                "idx": i,
                                "action": "SELL",
                                "amount": position,
                                "price": price,
                                "reason": "trailing",
                            }
                        )
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
                        trade_price = price * (1 + strategy_slippage_pct)
                        cost = balance * strength
                        commission = cost * strategy_commission_pct
                        total_cost = cost + commission
                        if total_cost > balance:
                            cost = balance / (1 + strategy_commission_pct)
                            commission = cost * strategy_commission_pct
                        total_cost = cost + commission
                        amount = cost / trade_price
                        if amount < self.min_trade_size:
                            idx += 1
                            continue
                        position += amount
                        balance -= total_cost
                        position_cost += total_cost
                        trades.append((i, "BUY", amount, trade_price, balance))
                        performance_logs.append(
                            {
                                "idx": i,
                                "action": "BUY",
                                "amount": amount,
                                "price": trade_price,
                                "commission": commission,
                                "slippage": trade_price - price,
                                "scale": strength,
                                "position_after": position,
                                "balance_after": balance,
                                "tp_pct": strategy_profit_threshold,
                                "ts_pct": strategy_trailing_stop,
                            }
                        )
                        trade_amounts.append(amount)
                        bought_total += amount
                        if price > highest_price:
                            highest_price = price
                    elif action == "SELL" and position > 0:
                        potential_profit = position * price - position_cost
                        strategy_profit_threshold = getattr(
                            strategy, "profit_threshold", self.profit_threshold
                        )
                        if trade_full_balance:
                            amount = position
                        elif (
                            strength < 0.5
                            and position_cost > 0
                            and strategy_profit_threshold is not None
                            and potential_profit / position_cost >= strategy_profit_threshold
                        ):
                            amount = position
                        else:
                            amount = position * strength
                        if amount > position:
                            amount = position
                        if amount < self.min_trade_size:
                            idx += 1
                            continue
                        trade_price = price * (1 - strategy_slippage_pct)
                        revenue = amount * trade_price
                        commission = revenue * strategy_commission_pct
                        sell_cost = (position_cost / position) * amount
                        pnl = revenue - commission - sell_cost
                        balance += sell_cost + pnl
                        # position_cost already reduced below
                        position -= amount
                        position_cost -= sell_cost
                        trades.append((i, "SELL", amount, trade_price, balance))
                        performance_logs.append(
                            {
                                "idx": i,
                                "action": "SELL",
                                "amount": amount,
                                "price": trade_price,
                                "commission": commission,
                                "slippage": price - trade_price,
                                "scale": strength,
                                "position_after": position,
                                "balance_after": balance,
                                "tp_pct": strategy_profit_threshold,
                                "ts_pct": strategy_trailing_stop,
                                "pnl": pnl,
                            }
                        )
                        trade_amounts.append(amount)
                        sold_total += amount
                        if position == 0:
                            highest_price = 0.0
                    idx += 1

            if position > 0:
                trade_price = prices[-1] * (1 - strategy_slippage_pct)
                revenue = position * trade_price
                commission = revenue * strategy_commission_pct
                pnl = revenue - commission - position_cost
                balance += position_cost + pnl
                trades.append((len(prices) - 1, "SELL", position, trade_price, balance))
                performance_logs.append(
                    {
                        "idx": len(prices) - 1,
                        "action": "SELL",
                        "amount": position,
                        "price": trade_price,
                        "commission": commission,
                        "slippage": prices[-1] - trade_price,
                        "reason": "final_close",
                        "pnl": pnl,
                        "position_after": 0.0,
                        "balance_after": balance,
                    }
                )
                trade_amounts.append(position)
                sold_total += position
                position = 0.0
                position_cost = 0.0
                highest_price = 0.0

            holding_value = position * prices[-1]
            final_value = balance + holding_value
            profit = final_value - 10000.0
            profit_pct = (profit / 10000.0) * 100

            strategy_trailing_stop = getattr(
                strategy, "trailing_stop_pct", self.trailing_stop_pct
            )
            strategy_profit_threshold = getattr(
                strategy, "profit_threshold", self.profit_threshold
            )
            if hasattr(strategy, "missed_summary"):
                miss_buy, miss_sell, miss_profit = strategy.missed_summary()
                miss_logs = getattr(strategy, "missed_opportunities", [])
            else:
                miss_buy = miss_sell = 0
                miss_profit = 0.0
                miss_logs = []

            trade_count = len(trade_amounts)
            avg_trade_size = sum(trade_amounts) / trade_count if trade_count else 0.0
            expected_profit = profit + miss_profit
            miss_count = len(miss_logs)

            results.append(
                {
                    "name": strategy.name,
                    "prices": prices,
                    "trades": trades,
                    "details": performance_logs,
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
                    "opportunities": miss_logs,
                    "missed_buy": miss_buy,
                    "missed_sell": miss_sell,
                    "missed_profit": miss_profit,
                    "expected_profit": expected_profit,
                    "trade_count": trade_count,
                    "avg_trade_size": avg_trade_size,
                    "missed_count": miss_count,
                }
            )
            self.logger.log(f"{strategy.name} profit: {profit:.2f}")

        return results
