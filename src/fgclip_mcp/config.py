"""Configuration management module."""

import os
import threading
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from .exceptions import ConfigurationError

class Config(BaseSettings):
    """FG-CLIP MCP server configuration."""

    # API configuration
    api_key: Optional[str] = Field(
        default=None, 
        validation_alias="MCP_API_KEY", 
        description="FG-CLIP API key"
    )
    api_base_url: str = Field(
        default="https://api.research.360.cn/models/interface",
        validation_alias="MCP_API_BASE_URL",
        description="Base URL for API"
    )
    api_timeout: float = Field(
        default=30.0,
        validation_alias="MCP_API_TIMEOUT",
        description="API request timeout (seconds)"
    )

    # Storage configuration
    storage_dir: Optional[str] = Field(
        default=None,
        validation_alias="MCP_STORAGE_DIR",
        description="Directory for storing embeddings"
    )
    max_items: Optional[int] = Field(
        default=None,
        validation_alias="MCP_EMBED_MAX_ITEMS",
        description="Maximum number of stored items"
    )

    # Logging configuration
    log_level: str = Field(
        default="INFO",
        validation_alias="MCP_LOG_LEVEL",
        description="Log level"
    )

    # Pydantic 2.x configuration
    model_config = {
        "env_file": None,  # do not load .env file
        "case_sensitive": False,
        "env_prefix": "",  # explicitly set empty env var prefix
        "populate_by_name": True,  # allow both field names and env var names
    }

    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v):
        """Validate API key."""
        if v is None:
            raise ConfigurationError("API key must not be empty")
        return v

    @field_validator('api_timeout')
    @classmethod
    def validate_api_timeout(cls, v):
        """Validate API timeout."""
        if v <= 0:
            raise ConfigurationError("API timeout must be greater than 0")
        return v

    @field_validator('max_items')
    @classmethod
    def validate_max_items(cls, v):
        """Validate maximum number of items."""
        if v is not None and v <= 0:
            raise ConfigurationError("max_items must be greater than 0")
        return v

    def get_storage_dir(self) -> str:
        """Get storage directory, or default if not set."""
        if self.storage_dir:
            return self.storage_dir
        return os.path.join(os.getcwd(), "embeddings")


# Thread-safe singleton implementation
_config_instance: Optional[Config] = None
_config_lock = threading.Lock()

def get_config(reload: bool = False) -> Config:
    """Get configuration instance (thread-safe).

    Args:
        reload: whether to reload configuration
    """
    global _config_instance
    
    with _config_lock:
        if _config_instance is None or reload:
            _config_instance = Config()
    
    return _config_instance