"""Service for retrieving price data and historical candles."""

from __future__ import annotations

import json
import random
import urllib.request
from typing import Iterable, List
import time


class DataService:
    """Retrieve price information from Binance or generate fallback data."""

    def fetch_price(self, symbol: str = "BTCUSDT") -> float:
        """Return the latest price for the given symbol."""
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        try:
            with urllib.request.urlopen(url) as resp:
                data = json.loads(resp.read().decode())
                return float(data["price"])
        except Exception:
            # Network might be disabled; return placeholder value
            return 0.0

    def get_historical_prices(self, symbol: str = "BTCUSDT", limit: int = 100) -> List[float]:
        """Return a list of closing prices for the given symbol."""
        url = (
            "https://api.binance.com/api/v3/klines"
            f"?symbol={symbol}&interval=1h&limit={limit}"
        )
        try:
            with urllib.request.urlopen(url) as resp:
                data = json.loads(resp.read().decode())
            return [float(item[4]) for item in data]
        except Exception:
            prices = [10000.0]
            for _ in range(limit - 1):
                prices.append(prices[-1] * (1 + random.uniform(-0.02, 0.02)))
            return prices

    def stream_prices(
        self,
        symbol: str = "BTCUSDT",
        interval: float = 300.0,
        limit: int | None = None,
    ) -> Iterable[float]:
        """Yield the latest price at the given interval."""
        count = 0
        while True:
            if limit is not None and count >= limit:
                break
            yield self.fetch_price(symbol)
            count += 1
            time.sleep(interval)
