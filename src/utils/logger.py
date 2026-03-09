"""
Centralized logging configuration for the governance system.

Provides structured logging with JSON formatting for production
and human-readable formatting for development environments.
"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging in production."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data

        return json.dumps(log_entry, default=str)


class GovernanceLogger:
    """
    Centralized logger factory for the governance system.

    Supports both JSON (production) and human-readable (development) output,
    with optional file rotation and component-scoped loggers.
    """

    _loggers: dict[str, logging.Logger] = {}

    @classmethod
    def get_logger(
        cls,
        name: str,
        level: str = "INFO",
        json_format: bool = False,
        log_file: Optional[str] = None,
        max_bytes: int = 10_485_760,
        backup_count: int = 5,
    ) -> logging.Logger:
        """
        Get or create a named logger with the specified configuration.

        Args:
            name: Logger name, typically the module path.
            level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            json_format: Use JSON formatting for structured log output.
            log_file: Optional file path for log file output with rotation.
            max_bytes: Maximum log file size before rotation (default 10MB).
            backup_count: Number of rotated log files to retain.

        Returns:
            Configured logging.Logger instance.
        """
        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.propagate = False

        if not logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))

            if json_format:
                console_handler.setFormatter(JSONFormatter())
            else:
                formatter = logging.Formatter(
                    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
                console_handler.setFormatter(formatter)

            logger.addHandler(console_handler)

            if log_file:
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)

                file_handler = logging.handlers.RotatingFileHandler(
                    filename=str(log_path),
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                )
                file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))

                if json_format:
                    file_handler.setFormatter(JSONFormatter())
                else:
                    file_formatter = logging.Formatter(
                        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )
                    file_handler.setFormatter(file_formatter)

                logger.addHandler(file_handler)

        cls._loggers[name] = logger
        return logger

    @classmethod
    def reset(cls) -> None:
        """Remove all cached loggers and their handlers."""
        for logger in cls._loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        cls._loggers.clear()


def get_logger(name: str, **kwargs) -> logging.Logger:
    """Convenience function to obtain a governance logger."""
    return GovernanceLogger.get_logger(name, **kwargs)
