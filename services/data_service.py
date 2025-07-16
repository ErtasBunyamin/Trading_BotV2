"""Service for retrieving price data from Binance public API."""


class DataService:
    """Retrieve price information from Binance."""

    def fetch_price(self, symbol: str = "BTCUSDT") -> float:
        """Return the latest price for the given symbol."""
        return 0.0
