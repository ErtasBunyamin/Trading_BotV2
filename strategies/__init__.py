"""Collection of trading strategies."""

from .rsi import RSIStrategy
from .macd import MACDStrategy
from .bollinger import BollingerStrategy
from .ma_cross import MACrossStrategy
from .random_strategy import RandomStrategy


__all__ = [
    "RSIStrategy",
    "MACDStrategy",
    "BollingerStrategy",
    "MACrossStrategy",
    "RandomStrategy",
]
