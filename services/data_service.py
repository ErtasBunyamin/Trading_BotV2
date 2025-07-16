"""Service for retrieving price data from Binance public API."""

from __future__ import annotations

import requests


class DataService:
    """Retrieve price information from Binance."""

    API_URL = "https://api.binance.com/api/v3/ticker/price"

    def fetch_price(self, symbol: str = "BTCUSDT") -> float:
        """Return the latest price for the given symbol.

        If the request fails, ``0.0`` is returned.
        """
        try:
            response = requests.get(self.API_URL, params={"symbol": symbol}, timeout=5)
            response.raise_for_status()
            data = response.json()
            return float(data.get("price", 0.0))
        except Exception:
            # In a real application we would log this error
            return 0.0


    def fetch_historical_prices(
        self, symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 200
    ) -> list[float]:
        """Return a list of historical closing prices.

        If the request fails, an empty list is returned.
        """
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        try:
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            return [float(item[4]) for item in data]  # close price
        except Exception:
            return []
