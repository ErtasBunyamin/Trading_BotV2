"""Service for retrieving price data and historical candles."""

from __future__ import annotations

import json
import random
import urllib.request
from typing import List


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

    def get_historical_prices(
        self, symbol: str = "BTCUSDT", limit: int | None = 100, interval: str = "1h"
    ) -> List[float]:
        """Return a list of closing prices for the given symbol.

        ``limit`` controls how many candles to retrieve. ``None`` attempts to
        fetch the maximum allowed by the API (1000).
        """
        base = "https://api.binance.com/api/v3/klines"
        if limit is None:
            url = f"{base}?symbol={symbol}&interval={interval}"
            fallback_len = 1000
        else:
            url = f"{base}?symbol={symbol}&interval={interval}&limit={limit}"
            fallback_len = limit
        try:
            with urllib.request.urlopen(url) as resp:
                data = json.loads(resp.read().decode())
            return [float(item[4]) for item in data]
        except Exception:
            if fallback_len is None:
                fallback_len = 1000
            prices = [10000.0]
            for _ in range(fallback_len - 1):
                prices.append(prices[-1] * (1 + random.uniform(-0.02, 0.02)))
            return prices
