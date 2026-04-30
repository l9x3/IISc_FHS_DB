"""
Logging configuration for the federated learning system.
"""

import logging
import os
from datetime import datetime


def get_logger(name: str, log_dir: str = "results", level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a named logger with console and optional file output.

    Parameters
    ----------
    name : str
        Logger name (usually the module's ``__name__``).
    log_dir : str
        Directory where log files are stored.  Pass ``""`` to disable
        file logging.
    level : int
        Logging level (default: ``logging.INFO``).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (optional)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"federated_{timestamp}.log")
        fh = logging.FileHandler(log_path)
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
