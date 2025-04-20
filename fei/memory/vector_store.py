"""
Vector store for semantic memory search in Fei.

This module provides a vector store implementation for semantic
memory search using embeddings.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime

from fei.utils.logging import get_logger
from fei.utils.config import Config
from fei.memory.embeddings import EmbeddingProvider

logger = get_logger(__name__)


class VectorStore:
    """
    Vector store for semantic memory search.
    
    This class provides a simple vector store implementation for
    semantic memory search using embeddings.
    """
    
    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        config: Optional[Config] = None,
        vector_dimension: int = 1536,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize the vector store.
        
        Args:
            embedding_provider: Provider for generating embeddings
            config: Configuration object
            vector_dimension: Dimension of embedding vectors
            similarity_threshold: Threshold for similarity search
        """
        self.config = config or Config()
        self.embedding_provider = embedding_provider
        self.vector_dimension = vector_dimension
        self.similarity_threshold = similarity_threshold
        
        # Storage for vectors and metadata
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
        # Path for persistent storage
        self.storage_path = self._get_storage_path()
        
        # Load existing vectors if available
        self._load_vectors()
    
    def _get_storage_path(self) -> str:
        """
        Get the path for vector storage.
        
        Returns:
            Path for vector storage
        """
        # Get storage path from config or use default
        storage_path = self.config.get("vector_store_path")
        if storage_path:
            return storage_path
        
        # Use default path in user's home directory
        home_dir = os.path.expanduser("~")
        fei_dir = os.path.join(home_dir, ".fei", "vectors")
        os.makedirs(fei_dir, exist_ok=True)
        return os.path.join(fei_dir, "vectors.json")
    
    def _load_vectors(self) -> None:
        """Load vectors from storage."""
        if not os.path.exists(self.storage_path):
            logger.info(f"Vector storage file not found at {self.storage_path}")
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            # Convert stored vectors back to numpy arrays
            for memory_id, vector_data in data.get("vectors", {}).items():
                self.vectors[memory_id] = np.array(vector_data)
            
            # Load metadata
            self.metadata = data.get("metadata", {})
            
            logger.info(f"Loaded {len(self.vectors)} vectors from storage")
        except Exception as e:
            logger.error(f"Error loading vectors: {e}")
    
    def _save_vectors(self) -> None:
        """Save vectors to storage."""
        try:
            # Convert numpy arrays to lists for JSON serialization
            vector_data = {}
            for memory_id, vector in self.vectors.items():
                vector_data[memory_id] = vector.tolist()
            
            data = {
                "vectors": vector_data,
                "metadata": self.metadata
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f)
            
            logger.debug(f"Saved {len(self.vectors)} vectors to storage")
        except Exception as e:
            logger.error(f"Error saving vectors: {e}")
    
    async def add_memory(
        self,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a memory to the vector store.
        
        Args:
            memory_id: Memory ID
            content: Memory content
            metadata: Optional metadata
            
        Returns:
            True if successful, False otherwise
        """
        if not self.embedding_provider:
            logger.error("No embedding provider available")
            return False
        
        try:
            # Generate embedding for content
            embedding = await self.embedding_provider.get_embedding(content)
            
            # Store embedding and metadata
            self.vectors[memory_id] = embedding
            self.metadata[memory_id] = metadata or {}
            
            # Add timestamp if not present
            if "timestamp" not in self.metadata[memory_id]:
                self.metadata[memory_id]["timestamp"] = datetime.now().isoformat()
            
            # Save to storage
            self._save_vectors()
            
            return True
        except Exception as e:
            logger.error(f"Error adding memory to vector store: {e}")
            return False
    
    def remove_memory(self, memory_id: str) -> bool:
        """
        Remove a memory from the vector store.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            True if successful, False if not found
        """
        if memory_id not in self.vectors:
            return False
        
        # Remove vector and metadata
        del self.vectors[memory_id]
        if memory_id in self.metadata:
            del self.metadata[memory_id]
        
        # Save to storage
        self._save_vectors()
        
        return True
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for memories similar to the query.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            threshold: Optional similarity threshold (overrides default)
            
        Returns:
            List of matching memories with similarity scores
        """
        if not self.embedding_provider:
            logger.error("No embedding provider available")
            return []
        
        if not self.vectors:
            logger.info("No vectors in store")
            return []
        
        try:
            # Generate embedding for query
            query_embedding = await self.embedding_provider.get_embedding(query)
            
            # Calculate similarity scores
            results = []
            for memory_id, vector in self.vectors.items():
                similarity = self._calculate_similarity(query_embedding, vector)
                
                # Apply threshold
                threshold_value = threshold or self.similarity_threshold
                if similarity >= threshold_value:
                    results.append({
                        "id": memory_id,
                        "similarity": float(similarity),
                        "metadata": self.metadata.get(memory_id, {})
                    })
            
            # Sort by similarity (descending)
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Limit results
            return results[:limit]
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (0-1)
        """
        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        # Calculate cosine similarity
        return float(np.dot(vec1_norm, vec2_norm))
    
    def get_memory_metadata(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a memory.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            Metadata or None if not found
        """
        return self.metadata.get(memory_id)
    
    def update_memory_metadata(
        self,
        memory_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Update metadata for a memory.
        
        Args:
            memory_id: Memory ID
            metadata: New metadata
            
        Returns:
            True if successful, False if not found
        """
        if memory_id not in self.metadata:
            return False
        
        # Update metadata
        self.metadata[memory_id].update(metadata)
        
        # Save to storage
        self._save_vectors()
        
        return True
    
    def clear(self) -> None:
        """Clear all vectors and metadata."""
        self.vectors = {}
        self.metadata = {}
        
        # Save to storage
        self._save_vectors()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Statistics dictionary
        """
        return {
            "memory_count": len(self.vectors),
            "vector_dimension": self.vector_dimension,
            "similarity_threshold": self.similarity_threshold,
            "storage_path": self.storage_path
        }
