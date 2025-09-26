"""MCP tools for FG-CLIP embedding services."""

import numpy as np
from copy import deepcopy
from typing import Any, Dict, List

from mcp.types import Tool

from .embedding_client import EmbeddingClient
from .exceptions import EmbeddingError, ValidationError, APIError
from .logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingTools:
    """MCP tools for FG-CLIP embedding services."""
    
    def __init__(self):
        """Initialize the embedding tools."""
        self.client = EmbeddingClient()
        self.result = {"success":False, "error_msg":""}
    
    
    def _extract_embeddings(self, api_result: Dict[str, Any], tool_name: str) -> List[List[float]]:
        """Extract embeddings array from API response with schema guard."""
        try:
            return api_result["data"]["embeddings"]["float"]
        except KeyError as e:
            logger.error("Unexpected API response schema for %s: %s", tool_name, api_result)
            raise APIError(
                f"Unexpected API response schema for {tool_name}",
                error_code="INVALID_API_RESPONSE",
                details={"tool_name": tool_name, "response": api_result, "missing_key": str(e)}
            ) from e
    
    def _save_embeddings(
        self,
        items: List[Any],
        embeddings: List[List[float]],
        *,
        model: str,
        input_type: str,
        storage,
    ) -> List[str]:
        """Save a batch of embeddings to storage with per-item error isolation."""
        if storage is None:
            logger.error("No storage provided, skipping embedding persistence")
            raise EmbeddingError("No storage provided, skipping embedding persistence", error_code="NO_STORAGE")
        
        saved_uris: List[str] = []
        for index, (item, embedding) in enumerate(zip(items, embeddings)):
            try:
                uri = storage.save_embedding(
                    content=item,
                    embedding=embedding,
                    model=model,
                    input_type=input_type,
                    metadata={f"{input_type}_index": index},
                )
                saved_uris.append(uri)
            except Exception as e:
                logger.error("Failed to save embedding for %s %d: %s", input_type, index, e)
                raise EmbeddingError(
                    f"Failed to save embeddings for {input_type} {index}",
                    error_code="SAVE_EMBEDDING_FAILED",
                    details={"input_type": input_type, "index": index, "original_error": str(e)}
                ) from e
        logger.info("Saved %d %s embeddings to storage", len(saved_uris), input_type)
        return saved_uris
    
    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors row-wise with zero-protection."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms
     
    def get_tools(self) -> List[Tool]:
        """Get the list of available tools."""
        return [
            Tool(
                name="text_embedding",
                description="Generate embeddings for text using FG-CLIP API",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "texts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of text strings to embed"
                        },
                        "model": {
                            "type": "string",
                            "default": "fg-clip",
                            "description": "Model to use for embedding"
                        }
                    },
                    "required": ["texts"]
                }
            ),
            Tool(
                name="image_embedding",
                description="Generate embeddings for images using FG-CLIP API",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "images": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of image URLs or base64 encoded images"
                        },
                        "model": {
                            "type": "string",
                            "default": "fg-clip",
                            "description": "Model to use for embedding"
                        }
                    },
                    "required": ["images"]
                }
            ),
            Tool(
                name="cosine_similarity",
                description="Compute cosine similarity(s) between two lists of vectors in range [-1, 1]. Supports pairwise or full matrix. Can use either direct vectors or resource URIs.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "vectors_a": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "number"}
                            },
                            "description": "First list of vectors (list[list[float]])"
                        },
                        "vectors_b": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "number"}
                            },
                            "description": "Second list of vectors (list[list[float]])"
                        },
                        "uris_a": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of embedding resource URIs for A"
                        },
                        "uris_b": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of embedding resource URIs for B"
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["pairwise", "matrix"],
                            "default": "pairwise",
                            "description": "pairwise: element-wise A[i] vs B[i]; matrix: all pairs A[i] vs B[j]"
                        }
                    },
                    "required": []
                }
            )            
        ]
    
    async def _call_embedding_tool(
        self, 
        arguments: Dict[str, Any], 
        input_type: str,
        items_key: str,
        items: List[str],
        model: str,
        storage=None, 
    ) -> dict:
        """Generic embedding tool call method."""
        cur_result = deepcopy(self.result)
        try:
            if not items:
                raise ValidationError(f"No {items_key} provided", error_code=f"MISSING_{items_key.upper()}")
            
            # Call corresponding API method based on input type
            if input_type == "text":
                result = await self.client.get_text_embeddings(texts=items, model=model)
            elif input_type == "image":
                result = await self.client.get_image_embeddings(images=items, model=model)
            else:
                raise ValidationError(f"Invalid input type: {input_type}", error_code="INVALID_INPUT_TYPE")
            
            embeddings = self._extract_embeddings(result, f"{input_type}_embedding")
            
            # Save embeddings to storage (if storage is provided)
            saved_uris = self._save_embeddings(items, embeddings, model=model, input_type=input_type, storage=storage)
            
            cur_result["saved_uris"] = saved_uris
            cur_result["success"] = True
            return cur_result
            
        except Exception as e:
            logger.error("Error in %s_embedding tool: %s", input_type, e)
            raise EmbeddingError(f"Error in {input_type}_embedding tool: {e}", error_code=f"{input_type.upper()}_EMBEDDING_ERROR") from e
    
    async def call_text_embedding(self, arguments: Dict[str, Any], storage=None) -> dict:
        """Call the text embedding tool."""
        texts = arguments.get("texts", [])
        model = arguments.get("model", "fg-clip")
        return await self._call_embedding_tool(
            arguments, "text", "texts", texts, model, storage
        )
    
    async def call_image_embedding(self, arguments: Dict[str, Any], storage=None) -> dict:
        """Call the image embedding tool."""
        images = arguments.get("images", [])
        model = arguments.get("model", "fg-clip")
        return await self._call_embedding_tool(
            arguments, "image", "images", images, model, storage
        )
    
    async def call_cosine_similarity(self, arguments: Dict[str, Any], storage=None) -> dict:
        """Call the cosine similarity tool."""
        cur_result = deepcopy(self.result)
        try:
            uris_a = arguments.get("uris_a", [])
            uris_b = arguments.get("uris_b", [])
            mode = arguments.get("mode", "pairwise")
            
            # Validate input parameters
            if not uris_a or not uris_b:
                error_msg = "Error: uris_a and uris_b must be provided"
                raise ValidationError(error_msg, error_code="MISSING_URIS")
            
            # Fetch embedding vectors from storage if URIs are provided
            def _fetch_embeddings_from_uris(uris):
                vectors = []
                for uri in uris:
                    embedding_data = storage.get_embedding(uri)
                    if embedding_data is None:
                        error_msg = f"Error: Embedding not found for URI: {uri}"
                        raise ValidationError(error_msg, error_code="EMBEDDING_NOT_FOUND")
                    vectors.append(embedding_data["embedding"])
                return vectors

            vectors_a, vectors_b = _fetch_embeddings_from_uris(uris_a ), _fetch_embeddings_from_uris(uris_b)
            
            # Convert to numpy arrays
            vectors_a = np.array(vectors_a, dtype=np.float32)
            vectors_b = np.array(vectors_b, dtype=np.float32)
            
            # Compute cosine similarity
            if mode == "pairwise":
                # Compute pairwise similarities
                if len(vectors_a) != len(vectors_b):
                    error_msg = "Error: vectors_a and vectors_b must have the same length for pairwise mode"
                    raise ValidationError(error_msg, error_code="INVALID_INPUT_PARAMETERS")
                
                # Normalize vectors
                vectors_a_norm = self._normalize(vectors_a)
                vectors_b_norm = self._normalize(vectors_b)
                
                # Calculate cosine similarities
                similarities = np.sum(vectors_a_norm * vectors_b_norm, axis=1)
                similarities = similarities.tolist()
                
                cur_result["similarities"] = similarities
                cur_result["shape"] = [len(similarities)]
                
            elif mode == "matrix":
                # Compute similarity matrix for all vector pairs
                vectors_a_norm = self._normalize(vectors_a)
                vectors_b_norm = self._normalize(vectors_b)
                
                # Calculate similarity matrix
                similarities_matrix = np.dot(vectors_a_norm, vectors_b_norm.T)
                similarities = similarities_matrix.tolist()
                
                cur_result["similarities"] = similarities
                cur_result["shape"] = [len(vectors_a), len(vectors_b)]
            
            else:
                error_msg = f"Error: Invalid mode '{mode}'. Must be 'pairwise' or 'matrix'"
                raise ValidationError(error_msg, error_code="INVALID_MODE")
            
            cur_result["success"] = True
            return cur_result
            
        except Exception as e:
            logger.error("Error in cosine_similarity tool: %s", e)
            raise EmbeddingError(f"Error in cosine_similarity tool: {e}", error_code="COSINE_SIMILARITY_ERROR") from e
