"""Collection of trading strategies."""

from .rsi import RSIStrategy
from .macd import MACDStrategy
from .bollinger import BollingerStrategy
from .ma_cross import MACrossStrategy
from .random_strategy import RandomStrategy
from .custom_strategy import CustomStrategy


__all__ = [
    "RSIStrategy",
    "MACDStrategy",
    "BollingerStrategy",
    "MACrossStrategy",
    "RandomStrategy",
    "CustomStrategy",
]
