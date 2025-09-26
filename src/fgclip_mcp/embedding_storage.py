"""Embedding storage module for MCP resources."""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import hashlib

from .config import get_config
from .exceptions import StorageError
from .logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingStorage:
    """Storage manager for embeddings as MCP resources."""
    
    def __init__(self, storage_dir: Optional[str] = None, max_items: Optional[int] = None) -> None:
        """Initialize embedding storage.
        
        Args:
            storage_dir: Directory to store embedding files. Defaults to config value
            max_items: Maximum number of embeddings to keep (oldest deleted first). If None, uses config value.
        """
        cfg = get_config()
        self.storage_dir = Path(storage_dir or cfg.get_storage_dir())
        self.storage_dir.mkdir(exist_ok=True)
        
        # Use config value if not provided
        self.max_items = max_items if max_items is not None else cfg.max_items
        
        # Index file that stores embedding metadata
        self.index_file = self.storage_dir / "index.json"
        self._load_index()
    
    def __enter__(self) -> "EmbeddingStorage":
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[object]) -> None:
        """Exit context manager."""
        if exc_type is not None:
            logger.error("Error in EmbeddingStorage context: %s", exc_val)
        # Add cleanup logic here if needed, e.g., persist index
        self._save_index()
    
    def _load_index(self) -> None:
        """Load the embedding index file."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.index = json.load(f)
            except Exception as e:
                self.index = {}
                logger.error("Failed to load index file: %s", e)
                raise StorageError("Failed to load index file: %s" % e, error_code="INDEX_LOAD_FAILED") from e
        else:
            self.index = {}
    
    def _save_index(self) -> None:
        """Save the embedding index file."""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Failed to save index file: %s", e)
            raise StorageError("Failed to save index file: %s" % e, error_code="INDEX_SAVE_FAILED") from e
    
    def _sorted_uris_by_created(self) -> List[str]:
        """Get URIs sorted by creation time ascending (oldest first)."""
        try:
            return [uri for uri, _ in sorted(self.index.items(), key=lambda kv: kv[1]["created_at"])]
        except Exception:
            return list(self.index.keys())
    
    def _generate_uri(self, content: str, model: str, input_type: str) -> str:
        """Generate a unique URI for the embedding."""
        # Use content hash and parameters to generate a unique identifier
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
        uri = f"embedding://{input_type}/{model}/{content_hash}"
        return uri
    
    def save_embedding(
        self, 
        content: str, 
        embedding: List[float], 
        model: str = "fg-clip",
        input_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Persist an embedding to storage.
        
        Args:
            content: Original content (text or image URL/base64)
            embedding: Embedding vector
            model: Model name used to generate the embedding
            input_type: Input type ("text" or "image")
            metadata: Optional additional metadata
            
        Returns:
            The generated embedding URI
        """
        uri = self._generate_uri(content, model, input_type)
        
        # Build embedding data payload
        embedding_data = {
            "uri": uri,
            "content": content,
            "embedding": embedding,
            "model": model,
            "input_type": input_type,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Save to file
        embedding_file = self.storage_dir / f"{uri.replace('://', '_').replace('/', '_')}.json"
        try:
            with open(embedding_file, 'w', encoding='utf-8') as f:
                json.dump(embedding_data, f, ensure_ascii=False, indent=2)
            
            # Update index
            self.index[uri] = {
                "file": str(embedding_file),
                "created_at": embedding_data["created_at"],
                "model": model,
                "input_type": input_type,
                "content_preview": content[:100] + "..." if len(content) > 100 else content
            }
            self._save_index()
            
            # Enforce count limit after saving
            self._enforce_limits()
            
            logger.info("Saved embedding with URI: %s", uri)
            return uri
            
        except Exception as e:
            logger.error("Failed to save embeddings: %s", e)
            raise RuntimeError("Failed to save embeddings: %s" % e) from e
    
    def get_embedding(self, uri: str) -> Optional[Dict[str, Any]]:
        """Get embedding data by URI.
        
        Args:
            uri: Embedding URI
            
        Returns:
            Embedding data dict, or None if not found
        """
        if uri not in self.index:
            return None
        
        try:
            embedding_file = Path(self.index[uri]["file"])
            if not embedding_file.exists():
                logger.error("Embedding file not found: %s", embedding_file)
                raise RuntimeError("Embedding file not found: %s" % embedding_file)
            
            with open(embedding_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error("Failed to load embedding %s: %s", uri, e)
            raise RuntimeError("Failed to load embedding %s: %s" % (uri, e)) from e
    
    def list_embeddings(self) -> List[Dict[str, Any]]:
        """List all saved embeddings.
        
        Returns:
            A list of embeddings containing URI and basic information
        """
        embeddings = []
        for uri, info in self.index.items():
            embeddings.append({
                "uri": uri,
                "name": f"Embedding ({info['input_type']})",
                "description": f"Embedding for {info['content_preview']} using {info['model']}",
                "created_at": info["created_at"],
                "model": info["model"],
                "input_type": info["input_type"]
            })
        
        # Sort by creation time
        embeddings.sort(key=lambda x: x["created_at"], reverse=True)
        return embeddings
    
    def delete_embedding(self, uri: str) -> bool:
        """Delete an embedding by URI.
        
        Args:
            uri: Embedding URI
            
        Returns:
            Whether deletion succeeded
        """
        if uri not in self.index:
            logger.error("Embedding not found: %s", uri)
            raise RuntimeError("Embedding not found: %s" % uri)
        
        try:
            # Delete file
            embedding_file = Path(self.index[uri]["file"])
            if embedding_file.exists():
                embedding_file.unlink()
            
            # Remove from index
            del self.index[uri]
            self._save_index()
            
            logger.info("Deleted embedding: %s", uri)
            return True
            
        except Exception as e:
            logger.error("Failed to delete embedding %s: %s", uri, e)
            raise RuntimeError("Failed to delete embedding %s: %s" % (uri, e)) from e
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            A dictionary with storage statistics
        """
        total_embeddings = len(self.index)
        text_embeddings = sum(1 for info in self.index.values() if info["input_type"] == "text")
        image_embeddings = sum(1 for info in self.index.values() if info["input_type"] == "image")
        
        return {
            "total_embeddings": total_embeddings,
            "text_embeddings": text_embeddings,
            "image_embeddings": image_embeddings,
            "max_items": self.max_items,
            "storage_directory": str(self.storage_dir)
        }

    def _delete_without_save(self, uri: str) -> None:
        """Delete embedding and update index in-memory; caller saves index."""
        if uri not in self.index:
            return
        embedding_file = Path(self.index[uri]["file"])
        try:
            if embedding_file.exists():
                embedding_file.unlink()
        except Exception as e:
            logger.warning("Failed to delete file for %s: %s", uri, e)
        finally:
            if uri in self.index:
                del self.index[uri]

    def _enforce_limits(self) -> None:
        """Enforce max_items by evicting oldest first."""
        if self.max_items is None or self.max_items < 0:
            return
        if len(self.index) <= self.max_items:
            return
        uris_by_age = self._sorted_uris_by_created()
        changed = False
        for uri in uris_by_age:
            if len(self.index) <= self.max_items:
                break
            self._delete_without_save(uri)
            changed = True
        if changed:
            self._save_index()
            logger.info("Storage count limit enforced. Now %d embeddings.", len(self.index))

    def cleanup(self) -> None:
        """Public entry to enforce storage limits on demand."""
        self._enforce_limits()
