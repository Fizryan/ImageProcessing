import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


class LoggingManager:
    def __init__(self):
        raise RuntimeError("LoggingManager is a static class")

    @staticmethod
    def setup_logging(
        name: Optional[str] = None,
        log_file_path="logs/app.log",
        level: int = logging.INFO,
    ) -> logging.Logger:
        logger: logging.Logger = logging.getLogger(name)

        if logger.hasHandlers():
            return logger

        logger.setLevel(logging.DEBUG)

        console_handler: logging.StreamHandler = logging.StreamHandler()
        console_handler.setLevel(level)

        formatter: logging.Formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file_path:
            try:
                Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
                MAX_BYTES: int = 5 * 1024 * 1024
                BACKUP_COUNT: int = 5
                file_handler: RotatingFileHandler = RotatingFileHandler(
                    filename=log_file_path,
                    maxBytes=MAX_BYTES,
                    backupCount=BACKUP_COUNT,
                    encoding="utf-8",
                )
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                print(f"Error setting up file handler: {e}")

        return logger


def fmt_bool(value: bool) -> str:
    return "Yes" if value else "No"
