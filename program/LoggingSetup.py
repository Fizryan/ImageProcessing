# LoggingSetup.py

import logging
from pathlib import Path


def setup_logger(
    name: str = None, log_file: str = None, level=logging.INFO
) -> logging.Logger:
    """
    Setup a logger with consistent formatting across the application.

    Args:
        name: Logger name (module name). If None, returns root logger
        log_file: Optional log file path
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if log_file:
        Path(log_file).parent.mkdir(exist_ok=True, parents=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("torchvision").setLevel(logging.WARNING)

    return logger


def fmt_bool(value: bool) -> str:
    """Format boolean as checkmark/cross symbol."""
    return "✓" if value else "✗"
