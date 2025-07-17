"""Dynamic hybrid trading strategy using multiple indicators.

This strategy aggregates signals from several indicators and adjusts
its decision threshold based on market regime and recent performance of
each indicator. It also factors in the current price location within a
lookback window and the short-term trend direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from .rsi import RSIStrategy
from .macd import MACDStrategy
from .bollinger import BollingerStrategy
from .ma_cross import MACrossStrategy


class DynamicHybridStrategy:
    """Generate signals with adaptive thresholds based on market regime."""

    name = "Dynamic Hybrid"

    def __init__(
        self,
        base_threshold: float = 0.12,
        volatile_threshold: float = 0.18,
        sideways_threshold: float = 0.08,
        lookback: int = 50,
        *,
        std_coef: float = 0.004,
        slope_threshold: float = 0.0002,
        location_weight: float = 0.2,
        trend_weight: float = 0.2,
        min_signal_distance: int = 2,
        confidence_threshold: float = 0.1,
        atr_period: int = 14,
        atr_multiplier: float = 0.5,
        volume_period: int = 20,
        volume_multiplier: float = 1.3,
        risk_atr_scale: float = 1.0,
        drawdown_lookback: int = 3,
        drawdown_penalty: float = 0.5,
        winrate_window: int = 20,
        winrate_target: float = 0.55,
        ema_momentum_period: int = 20,
        momentum_weight: float = 0.1,
        session_thresholds: Dict[Tuple[int, int], float] | None = None,
        atr_stop_mult: float = 1.5,
        atr_profit_mult: float = 2.0,
        log_decisions: bool = True,
    ) -> None:
        self.indicator_winrates: Dict[str, float] = {
            "RSI": 0.5,
            "MACD": 0.5,
            "Bollinger": 0.5,
            "MA Cross": 0.5,
        }
        self.market_regime = "trend"
        self.base_threshold = base_threshold
        self.volatile_threshold = volatile_threshold
        self.sideways_threshold = sideways_threshold
        self.lookback = lookback
        self.std_coef = std_coef
        self.slope_threshold = slope_threshold
        self.location_weight = location_weight
        self.trend_weight = trend_weight
        self.min_signal_distance = min_signal_distance
        self.confidence_threshold = confidence_threshold

        # --- Advanced parameters ---
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.volume_period = volume_period
        self.volume_multiplier = volume_multiplier
        self.risk_atr_scale = risk_atr_scale
        self.drawdown_lookback = drawdown_lookback
        self.drawdown_penalty = drawdown_penalty
        self.winrate_window = winrate_window
        self.winrate_target = winrate_target
        self.ema_momentum_period = ema_momentum_period
        self.momentum_weight = momentum_weight
        self.session_thresholds = session_thresholds or {}
        self.atr_stop_mult = atr_stop_mult
        self.atr_profit_mult = atr_profit_mult
        self.log_decisions = log_decisions

        # Runtime state
        self.loss_streak = 0
        self._atr_values: List[float] = []
        self.decision_log: List[Tuple[int, str]] = []

        # Sub-strategies providing raw signals
        self._strategies = [
            RSIStrategy(),
            MACDStrategy(),
            BollingerStrategy(),
            MACrossStrategy(),
        ]

    def before_run(self, prices: List[float]) -> None:
        """Initialize ATR based stop levels before simulation starts."""
        base_atr = self._average_true_range(prices[-self.atr_period :])
        if base_atr and prices:
            last_price = prices[-1]
            self.trailing_stop_pct = (base_atr * self.atr_stop_mult) / last_price
            self.profit_threshold = (base_atr * self.atr_profit_mult) / last_price

    def update_winrates(self, trade_logs: List[Dict]) -> None:
        """Update indicator win rates using the last 20 trades."""
        for ind in self.indicator_winrates:
            trades = [t for t in trade_logs if t["indicator"] == ind][-20:]
            if trades:
                win = sum(1 for t in trades if t["pnl"] > 0)
                self.indicator_winrates[ind] = win / len(trades)
            else:
                self.indicator_winrates[ind] = 0.5

    def detect_market_regime(self, prices: List[float]) -> None:
        """Classify the market regime using EMA slope and volatility."""
        ema = pd.Series(prices).ewm(span=self.lookback).mean()
        window = prices[-self.lookback :]
        std = np.std(window)
        if len(ema) >= self.lookback:
            start_idx = -self.lookback
            length = self.lookback
        else:
            start_idx = 0
            length = max(1, len(ema) - 1)
        slope = (ema.iloc[-1] - ema.iloc[start_idx]) / length
        mean_price = np.mean(window)
        if std > self.std_coef * mean_price:
            self.market_regime = "volatile"
        elif abs(slope) < self.slope_threshold:
            self.market_regime = "sideways"
        else:
            self.market_regime = "trend"

    def dynamic_voting(self, indicator_signals: Dict[str, Tuple[str, float]]) -> float:
        """Vote on the final score using indicator win rates."""
        score = 0.0
        total_weight = sum(self.indicator_winrates.values())
        for ind, (action, strength) in indicator_signals.items():
            if ind not in self.indicator_winrates:
                # Register unseen indicator with a neutral win rate
                self.indicator_winrates[ind] = 0.5
                total_weight = sum(self.indicator_winrates.values())
            weight = (
                self.indicator_winrates[ind] / total_weight if total_weight else 0.25
            )
            if action == "BUY":
                score += strength * weight
            elif action == "SELL":
                score -= strength * weight
        return score

    def get_threshold(self) -> float:
        """Return the buy/sell threshold for the current market regime."""
        if self.market_regime == "trend":
            return self.base_threshold
        if self.market_regime == "volatile":
            return self.volatile_threshold
        return self.sideways_threshold

    def _trend_score(self, window: List[float]) -> float:
        """Return normalized trend strength for the given price window."""
        n = len(window)
        if n < 2:
            return 0.0
        xs = list(range(n))
        mean_x = sum(xs) / n
        mean_y = sum(window) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, window))
        den = sum((x - mean_x) ** 2 for x in xs)
        slope = num / den if den != 0 else 0.0
        trend = slope * n / window[-1]
        return max(-1.0, min(1.0, trend))

    def _average_true_range(self, closes: List[float]) -> float:
        """Approximate ATR using close-to-close changes."""
        if len(closes) < 2:
            return 0.0
        diffs = [abs(closes[i] - closes[i - 1]) for i in range(1, len(closes))]
        return float(np.mean(diffs[-self.atr_period :]))

    def _session_factor(self, idx: int) -> float:
        """Return threshold multiplier depending on time of day."""
        if not self.session_thresholds:
            return 1.0
        # 5 minute candles -> 12 per hour
        hour = (idx // 12) % 24
        for (start, end), factor in self.session_thresholds.items():
            if start <= hour < end:
                return factor
        return 1.0

    def _update_drawdown(self, trade_logs: List[Dict]) -> None:
        """Update consecutive loss streak for risk adjustment."""
        if not trade_logs:
            return
        last_pnl = trade_logs[-1].get("pnl", 0)
        if last_pnl <= 0:
            self.loss_streak += 1
        else:
            self.loss_streak = 0

    def generate_signals(
        self,
        prices: List[float],
        strat_signals: List[Tuple[int, str, float, str]] | None = None,
        trade_logs: List[Dict] | None = None,
        volumes: List[float] | None = None,
    ) -> List[Tuple[int, str, float]]:
        """Aggregate indicator signals and return trading decisions."""

        # When no indicator signals are supplied, gather them from the default
        # sub-strategies so this method is compatible with ``Simulation`` which
        # only passes ``prices``.
        if strat_signals is None:
            strat_signals = []
            for strat in self._strategies:
                for idx, action, strength in strat.generate_signals(prices):
                    strat_signals.append((idx, action, strength, strat.name))

        # Derive simple trade logs when not provided so indicator win rates can
        # still be updated. The outcome of each signal is evaluated on the next
        # bar only as a lightweight heuristic.
        if trade_logs is None:
            trade_logs = []
            signals_by_indicator: Dict[str, List[Tuple[int, str]]] = {}
            for idx, action, _strength, ind in strat_signals:
                signals_by_indicator.setdefault(ind, []).append((idx, action))
            for ind, sigs in signals_by_indicator.items():
                for idx, action in sigs[-20:]:
                    if idx + 1 < len(prices):
                        diff = prices[idx + 1] - prices[idx]
                        pnl = diff if action == "BUY" else -diff
                        trade_logs.append({"indicator": ind, "pnl": pnl})

        self.update_winrates(trade_logs)
        self._update_drawdown(trade_logs)

        if self.trailing_stop_pct is None or self.profit_threshold is None:
            base_atr = self._average_true_range(prices[-self.atr_period :])
            if base_atr and prices:
                last_price = prices[-1]
                self.trailing_stop_pct = (base_atr * self.atr_stop_mult) / last_price
                self.profit_threshold = (base_atr * self.atr_profit_mult) / last_price

        # Organize signals per bar as indicator -> (action, strength)
        signals_by_idx: Dict[int, Dict[str, Tuple[str, float]]] = {}
        for idx, action, strength, ind in strat_signals:
            signals_by_idx.setdefault(idx, {})[ind] = (action, strength)

        final_signals: List[Tuple[int, str, float]] = []
        last_idx = -self.min_signal_distance

        overall_trades = trade_logs[-self.winrate_window :]
        overall_win = (
            sum(1 for t in overall_trades if t["pnl"] > 0) / len(overall_trades)
            if overall_trades
            else 1.0
        )

        for idx in range(len(prices)):
            indicator_map = signals_by_idx.get(idx)
            if not indicator_map:
                continue

            # ATR and volume based filters
            atr = self._average_true_range(prices[max(0, idx - self.atr_period) : idx + 1])
            self._atr_values.append(atr)
            atr_allowed = True
            if len(self._atr_values) >= self.atr_period:
                atr_avg = float(np.mean(self._atr_values[-self.atr_period :]))
                atr_allowed = atr >= atr_avg * self.atr_multiplier
            else:
                atr_avg = atr

            volume_allowed = True
            if volumes is not None and idx >= self.volume_period:
                vol_avg = float(np.mean(volumes[idx - self.volume_period + 1 : idx + 1]))
                volume_allowed = volumes[idx] >= vol_avg * self.volume_multiplier
            if not atr_allowed or not volume_allowed:
                if self.log_decisions:
                    reason = "ATR düşük" if not atr_allowed else "Hacim düşük"
                    if not atr_allowed and not volume_allowed:
                        reason = "ATR & Hacim düşük"
                    self.decision_log.append((idx, reason))
                continue

            # Update stop levels dynamically with current ATR
            if prices[idx] > 0:
                self.trailing_stop_pct = (atr * self.atr_stop_mult) / prices[idx]
                self.profit_threshold = (atr * self.atr_profit_mult) / prices[idx]

            self.detect_market_regime(prices[: idx + 1])
            score = self.dynamic_voting(indicator_map)

            # Price location contribution within lookback window
            start = max(0, idx - self.lookback + 1)
            window = prices[start : idx + 1]
            low = min(window)
            high = max(window)
            if high != low:
                pos = (prices[idx] - low) / (high - low)
                score += (0.5 - pos) * 2 * self.location_weight

            # Trend contribution
            if self.trend_weight:
                score += self._trend_score(window) * self.trend_weight

            # Momentum & EMA filter
            if self.momentum_weight and idx >= self.ema_momentum_period:
                ema = (
                    pd.Series(prices[: idx + 1])
                    .ewm(span=self.ema_momentum_period)
                    .mean()
                    .iloc[-1]
                )
                high_mom = max(window[-self.ema_momentum_period :])
                near_high = high_mom != low and prices[idx] >= high_mom * 0.97
                if prices[idx] >= ema and near_high:
                    score += self.momentum_weight
                elif prices[idx] < ema:
                    score -= self.momentum_weight

            threshold = self.get_threshold()
            dynamic_conf = self.confidence_threshold
            dynamic_dist = self.min_signal_distance
            # adapt parameters when overall win rate drops
            if overall_trades and overall_win < self.winrate_target:
                delta = self.winrate_target - overall_win
                threshold *= 1 + delta
                dynamic_conf += delta * 0.5
                dynamic_dist = int(self.min_signal_distance + delta * 2)
            threshold *= self._session_factor(idx)

            if abs(score) < dynamic_conf:
                if self.log_decisions:
                    self.decision_log.append((idx, "confidence"))
                continue
            if idx - last_idx < dynamic_dist:
                if self.log_decisions:
                    self.decision_log.append((idx, "min_distance"))
                continue

            # Risk based strength scaling
            risk_scale = 1.0
            if atr_avg > 0:
                volatility_factor = atr / atr_avg
                risk_scale /= max(1.0, volatility_factor ** self.risk_atr_scale)
            if self.loss_streak > self.drawdown_lookback:
                risk_scale *= self.drawdown_penalty ** (self.loss_streak - self.drawdown_lookback)

            if score >= threshold:
                strength = min(1.0, abs(score) * risk_scale)
                final_signals.append((idx, "BUY", strength))
                last_idx = idx
            elif score <= -threshold:
                strength = min(1.0, abs(score) * risk_scale)
                final_signals.append((idx, "SELL", strength))
                last_idx = idx
            else:
                if self.log_decisions:
                    self.decision_log.append((idx, "Score düşük"))

        return final_signals


# Example usage
# strategy = DynamicHybridStrategy()
# signals = strategy.generate_signals(prices, volumes=volumes)
# print("Decision log", strategy.decision_log[:5])
