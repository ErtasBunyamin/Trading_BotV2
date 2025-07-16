"""Entry point for the trading bot simulator."""

from services.data_service import DataService
from services.simulation import Simulation
from services.logger import Logger


def main() -> None:
    """Run the application."""
    data_service = DataService()
    logger = Logger()
    simulation = Simulation(data_service, logger)

    logger.log("Application started")
    simulation.run()


if __name__ == "__main__":
    main()
