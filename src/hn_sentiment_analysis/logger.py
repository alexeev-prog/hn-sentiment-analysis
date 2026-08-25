import logging
import logging.handlers
import sys
from pathlib import Path

from hn_sentiment_analysis.config import settings


class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[41m",
        "RESET": "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        original_msg = super().format(record)
        return f"{color}{original_msg}{reset}"


def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    if settings.log_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)

        if sys.stdout.isatty():
            console_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            console_formatter = ColorFormatter(
                console_format, datefmt="%Y-%m-%d %H:%M:%S"
            )
        else:
            console_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            console_formatter = logging.Formatter(
                console_format, datefmt="%Y-%m-%d %H:%M:%S"
            )

        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    if settings.log_file:
        file_path = log_dir / settings.log_file
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)

        if settings.log_format == "json":
            file_format = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "module": "%(module)s", "function": "%(funcName)s", "line": %(lineno)d, "message": "%(message)s"}'
        else:
            file_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(lineno)d | %(funcName)s | %(message)s"

        file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


logger = setup_logging()
