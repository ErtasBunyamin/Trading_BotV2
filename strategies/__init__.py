"""Collection of built-in trading strategies."""

from .rsi_strategy import RSIStrategy
from .macd_strategy import MACDStrategy
from .bollinger_strategy import BollingerStrategy
from .ma_cross_strategy import MACrossStrategy
from .custom_strategy import CustomStrategy

__all__ = [
    "RSIStrategy",
    "MACDStrategy",
    "BollingerStrategy",
    "MACrossStrategy",
    "CustomStrategy",
]
