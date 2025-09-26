"""MCP server for FG-CLIP embedding services."""

import logging
from typing import Any, Dict, List
import json


from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    Tool,
    Resource
)

from .tools import EmbeddingTools
from .embedding_storage import EmbeddingStorage
from .exceptions import FGClipMCPError

from .logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingMCPServer:
    """MCP server for FG-CLIP embedding services."""
    def __init__(self):
        """Initialize the MCP server."""
        self.server = Server("embedding-mcp-server")
        self.embedding_tools = EmbeddingTools()
        self.embedding_storage = EmbeddingStorage()
        self._register_handlers()

    def _register_handlers(self):
        """Register MCP server handlers."""
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """Handle list tools request."""
            tools = self.embedding_tools.get_tools()
            return tools

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Handle tool call request."""
            logger.info("Calling tool: %s with arguments: %s", name, arguments)
            if name == "text_embedding":
                return await self.embedding_tools.call_text_embedding(arguments, self.embedding_storage)
            elif name == "image_embedding":
                return await self.embedding_tools.call_image_embedding(arguments, self.embedding_storage)
            elif name == "cosine_similarity":
                return await self.embedding_tools.call_cosine_similarity(arguments, self.embedding_storage)
            else:
                logger.error("Unknown tool: %s", name)
                raise FGClipMCPError(f"Unknown tool: {name}", error_code="UNKNOWN_TOOL")

        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            """Handle list resources request."""
            embeddings = self.embedding_storage.list_embeddings()
            resources = []
            for embedding in embeddings:
                resources.append(Resource(
                    uri=embedding["uri"],
                    name=embedding["name"],
                    description=embedding["description"],
                    mimeType="application/json"
                ))    
            # Add storage statistics as a special resource
            stats = self.embedding_storage.get_storage_stats()
            description = (
                f"Storage statistics: {stats['total_embeddings']} embeddings "
                f"({stats['text_embeddings']} text, {stats['image_embeddings']} image)"
            )
            resources.append(Resource(
                uri="embedding://stats",
                name="Embedding Storage Statistics",
                description=description,
                mimeType="application/json"
            ))
            return resources

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Handle read resource request."""
            logger.info("Reading resource: %s", uri)
            if str(uri) == "embedding://stats":
                # Return storage statistics
                stats = self.embedding_storage.get_storage_stats()
                return json.dumps(stats, ensure_ascii=False, indent=2)
            else:
                # Return embedding data
                embedding_data = self.embedding_storage.get_embedding(str(uri))
                if embedding_data is None:
                    logger.error("Resource not found: %s", uri)
                    raise FGClipMCPError(f"Resource not found: {uri}", error_code="RESOURCE_NOT_FOUND")
                return json.dumps(embedding_data, ensure_ascii=False, indent=2)

async def run_stdio_server():
    """Run the MCP server using stdio transport."""
    try:
        server = EmbeddingMCPServer()
        # Initialize the server
        init_options = InitializationOptions(
            server_name="embedding-mcp-server",
            server_version="0.1.2",
            capabilities={
                "tools": {},
                "resources": {}
            }
        )
        logger.info("Starting stdio MCP server...")
        async with stdio_server() as (read, write):
            logger.info("Stdio transport established, running server...")
            await server.server.run(
                read,
                write,
                init_options
            )
    except Exception as e:
        logger.error("Error in run_stdio_server: %s", e)
        raise FGClipMCPError(f"Error in run_stdio_server: {e}", error_code="SERVER_ERROR") from e
