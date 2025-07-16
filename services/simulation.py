"""Basic trading simulation logic."""

from services.data_service import DataService
from services.logger import Logger


class Simulation:
    """Run a simple price polling simulation."""

    def __init__(self, data_service: DataService, logger: Logger) -> None:
        self.data_service = data_service
        self.logger = logger

    def run(self) -> None:
        """Fetch the current price and log it."""
        price = self.data_service.fetch_price()
        self.logger.log(f"Current BTC price: {price}")
