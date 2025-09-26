"""MCP Server for FG-CLIP embedding services."""

import sys
import asyncio

from .server import run_stdio_server
from .logging_config import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)


def main():
    try:
        logger.info("Starting stdio server")
        asyncio.run(run_stdio_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error("Server error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
