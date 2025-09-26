"""Unified logging configuration module."""

import logging
import sys
import os
from typing import Optional


def setup_logging(level: Optional[str] = None) -> None:
    """Set up unified logging configuration."""
    
    # Get log level (only from env var to avoid config load at import time)
    log_level = level or os.getenv("MCP_LOG_LEVEL", "INFO")
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Set log levels for specific modules
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("mcp").setLevel(logging.INFO)
    
    # Record configuration info
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized, level: %s", log_level)


def get_logger(name: str) -> logging.Logger:
    """Get logger by name."""
    return logging.getLogger(name)
