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
    CustomStrategy,
    DynamicHybridStrategy,
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
        CustomStrategy(),
        DynamicHybridStrategy(),
    ]
    # Request the maximum available candle history so the GUI can display
    # all available bars without limitation.
    simulation = Simulation(
        data_service,
        logger,
        strategies,
        price_limit=None,
        full_balance=True,
        commission_pct=0.001,
        slippage_pct=0.0005,
        min_trade_size=0.0001,
    )

    logger.log("Application started")
    results = simulation.run()

    app = TradingApp(results)
    app.run()

if __name__ == "__main__":
    main()
