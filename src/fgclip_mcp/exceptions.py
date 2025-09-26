"""Custom exception classes."""

from typing import Any, Dict, Optional


class FGClipMCPError(Exception):
    """Base exception for FG-CLIP MCP server."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ConfigurationError(FGClipMCPError):
    """Configuration error."""
    pass


class APIError(FGClipMCPError):
    """API call error."""
    pass


class StorageError(FGClipMCPError):
    """Storage operation error."""
    pass


class ValidationError(FGClipMCPError):
    """Input validation error."""
    pass


class EmbeddingError(FGClipMCPError):
    """Embedding processing error."""
    pass
