"""FG-CLIP API client for embedding services."""
from typing import List, Dict, Any

import httpx

from .config import get_config
from .logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingClient:
    """Client for FG-CLIP embedding API."""
    
    def __init__(self, timeout: float = None):
        cfg = get_config()
        if not cfg.api_key:
            raise ValueError("MCP_API_KEY is not set. Please set environment variable MCP_API_KEY to use the embedding API.")
        self.base_url = cfg.api_base_url
        self.headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}"
        }
        self.timeout = timeout or cfg.api_timeout
    
    async def _post_embeddings(self, *, input_type: str, items_key: str, items: List[str], model: str) -> Dict[str, Any]:
        """Reusable HTTP POST to fetch embeddings for given input type."""
        payload = {
            "model": model,
            "input_type": input_type,
            items_key: items,
        }
        logger.info("Requesting %s embeddings for %d %s", input_type, len(items), items_key)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.base_url,
                headers={**self.headers},
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
    
    async def get_text_embeddings(
        self, 
        texts: List[str], 
        model: str = "fg-clip",
    ) -> Dict[str, Any]:
        """Get text embeddings from FG-CLIP API.
        
        Args:
            texts: List of text strings to embed
            model: Model name to use for embedding
            
        Returns:
            API response containing embeddings
            
        Raises:
            httpx.HTTPError: If API request fails
        """
        return await self._post_embeddings(input_type="text", items_key="texts", items=texts, model=model)
    
    async def get_image_embeddings(
        self, 
        images: List[str], 
        model: str = "fg-clip",
    ) -> Dict[str, Any]:
        """Get image embeddings from FG-CLIP API.
        
        Args:
            images: List of image URLs or base64 encoded images
            model: Model name to use for embedding
            
        Returns:
            API response containing embeddings
            
        Raises:
            httpx.HTTPError: If API request fails
        """
        return await self._post_embeddings(input_type="image", items_key="images", items=images, model=model)