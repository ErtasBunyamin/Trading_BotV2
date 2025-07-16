"""Entry point for the trading bot simulator."""

from services.data_service import DataService
from services.simulation import Simulation
from services.logger import Logger
from services.gui import TradingApp
from strategies import (
    RSIStrategy,
    MACDStrategy,
    BollingerStrategy,
    MACrossStrategy,
    RandomStrategy,
)


def main() -> None:
    """Run the application."""
    data_service = DataService()
    logger = Logger()
    strategies = [
        RSIStrategy(),
        MACDStrategy(),
        BollingerStrategy(),
        MACrossStrategy(),
        RandomStrategy(),
    ]
    simulation = Simulation(data_service, logger, strategies)

    logger.log("Application started")
    results = simulation.run()

    app = TradingApp(results)
    app.run()


if __name__ == "__main__":
    main()
