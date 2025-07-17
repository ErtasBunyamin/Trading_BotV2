"""Normalized gradient threshold trading strategy."""

from __future__ import annotations

from typing import List, Tuple
import statistics


class GradientStrategy:
    """Trade based on the slope of normalized prices.

    Prices are normalized using a rolling z-score. The gradient of the
    normalized series determines whether to open or close a position.
    The ``threshold`` is automatically increased when volatility exceeds
    ``volatility_threshold`` to avoid trading during noisy periods.
    """

    name = "Gradient"

    def __init__(
        self,
        window: int = 20,
        threshold: float = 0.01,
        volatility_window: int | None = None,
        volatility_threshold: float = 0.02,
        profit_threshold: float | None = None,
        trailing_stop_pct: float | None = None,
    ) -> None:
        self.window = window
        self.threshold = threshold
        self.volatility_window = volatility_window or window
        self.volatility_threshold = volatility_threshold
        self.profit_threshold = profit_threshold
        self.trailing_stop_pct = trailing_stop_pct
        self.trade_log: List[Tuple[int, str, float]] = []
        self.missed_opportunities: List[Tuple[int, float, str, float]] = []

    def _zscore(self, series: List[float]) -> float:
        mean = statistics.mean(series)
        std = statistics.stdev(series) if len(series) > 1 else 0.0
        if std == 0:
            return 0.0
        return (series[-1] - mean) / std

    def generate_signals(self, prices: List[float]) -> List[Tuple[int, str, float]]:
        signals: List[Tuple[int, str, float]] = []
        if not prices:
            return signals
        norm_prices: List[float] = []
        for i, _p in enumerate(prices):
            start = max(0, i - self.window + 1)
            window = prices[start : i + 1]
            norm = self._zscore(window)
            norm_prices.append(norm)
        position = 0
        for i in range(1, len(prices)):
            start = max(0, i - self.volatility_window + 1)
            vol_window = prices[start : i + 1]
            vol = statistics.stdev(vol_window) if len(vol_window) > 1 else 0.0
            thr = self.threshold
            if vol > self.volatility_threshold:
                thr *= vol / self.volatility_threshold
            grad = norm_prices[i] - norm_prices[i - 1]
            action: str | None = None
            strength = min(1.0, abs(grad) / thr) if thr else 1.0
            if grad > thr and position <= 0:
                action = "BUY"
                position = 1
            elif grad < -thr and position >= 0:
                action = "SELL"
                position = -1
            elif position == 1 and grad < -thr:
                action = "SELL"
                position = 0
            elif position == -1 and grad > thr:
                action = "BUY"
                position = 0
            if action:
                signals.append((i, action, strength))
                self.trade_log.append((i, action, prices[i]))
            else:
                if abs(grad) > thr and vol > self.volatility_threshold:
                    next_price = prices[i + 1] if i + 1 < len(prices) else prices[i]
                    pot = next_price - prices[i] if grad > 0 else prices[i] - next_price
                    missed_action = "BUY" if grad > 0 else "SELL"
                    self.missed_opportunities.append((i, prices[i], missed_action, pot))
        return signals

    def missed_summary(self) -> Tuple[int, int, float]:
        buy = sum(1 for *_idx, act, _ in self.missed_opportunities if act == "BUY")
        sell = sum(1 for *_idx, act, _ in self.missed_opportunities if act == "SELL")
        profit = sum(p for *_idx, p in self.missed_opportunities)
        return buy, sell, profit

