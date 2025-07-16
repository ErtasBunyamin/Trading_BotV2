"""Simple logging facility used across the application."""

import logging


class Logger:
    """Wrapper around :mod:`logging` for ease of use."""

    def __init__(self) -> None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
        self._logger = logging.getLogger("trading_bot")

    def log(self, message: str) -> None:
        """Log the provided message."""
        self._logger.info(message)
