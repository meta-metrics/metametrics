import logging
import os
import sys
import threading
from typing import Optional

# Global configurations for logger setup
_logger_lock = threading.RLock()
_handler: Optional["logging.Handler"] = None
_log_level: "logging._Level" = logging.INFO

def _determine_log_level() -> "logging._Level":
    """
    Retrieves the log level from the environment variable, if available, or defaults to INFO.
    """
    level_from_env = os.getenv("METAMETRICS_LOG_LEVEL", None)
    if level_from_env and level_from_env.upper() in logging._nameToLevel:
        return logging._nameToLevel[level_from_env.upper()]
    elif level_from_env:
        raise ValueError(f"Unrecognized logging level: {level_from_env}")
    return _log_level

def _get_main_module_name() -> str:
    return __name__.split(".")[0]

def _initialize_root_logger() -> "logging.Logger":
    return logging.getLogger(_get_main_module_name())

def _setup_logger() -> None:
    """
    Sets up a stream handler with a specific format for the root logger.
    """
    global _handler
    with _logger_lock:
        if _handler is not None:
            return

        # Define log format
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Set up handler to output to stdout
        _handler = logging.StreamHandler(sys.stdout)
        _handler.setFormatter(formatter)

        # Attach handler and set level
        root_logger = _initialize_root_logger()
        root_logger.addHandler(_handler)
        root_logger.setLevel(_determine_log_level())
        root_logger.propagate = False

def get_logger(logger_name: Optional[str] = None) -> "logging.Logger":
    """
    Generates a logger with the provided name, or defaults to the root logger's name.
    """
    if logger_name is None:
        logger_name = _get_main_module_name()

    _setup_logger()
    return logging.getLogger(logger_name)
