"""Embedding model utilities."""

import hashlib
import json
from pathlib import Path
from typing import List, Optional
import numpy as np
from openai import OpenAI

from config import Config


class EmbeddingGenerator:
    """Generate embeddings using OpenAI API."""
    
    def __init__(self, model: str = None):
        """
        Initialize embedding generator.
        
        Args:
            model: Embedding model name (defaults to config)
        """
        self.model = model or Config.RAG_EMBEDDING_MODEL
        
        # Require OpenAI API key for embeddings
        openai_key = Config.OPENAI_API_KEY
        if not openai_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found in environment variables. "
                "Embeddings require an OpenAI API key. Please set OPENAI_API_KEY in your .env file."
            )
        
        try:
            self.client = OpenAI(api_key=openai_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {e}")
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        """Generate embeddings using OpenAI client with batching for large requests."""
        # OpenAI has a limit of ~300k tokens per request
        # Process in smaller batches to avoid exceeding the limit
        # Estimate: ~500 tokens per document on average, so ~50-100 docs per batch is safe
        batch_size = 50  # Conservative batch size
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch_texts
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                # If batch is still too large, try smaller batches
                if "max_tokens_per_request" in str(e) or "max_tokens" in str(e).lower():
                    # Try even smaller batches
                    smaller_batch_size = 10
                    for j in range(0, len(batch_texts), smaller_batch_size):
                        smaller_batch = batch_texts[j:j + smaller_batch_size]
                        try:
                            response = self.client.embeddings.create(
                                model=self.model,
                                input=smaller_batch
                            )
                            all_embeddings.extend([item.embedding for item in response.data])
                        except Exception as e2:
                            raise RuntimeError(
                                f"Failed to generate embeddings for batch {i//batch_size + 1}, "
                                f"sub-batch {j//smaller_batch_size + 1}: {e2}"
                            )
                else:
                    raise RuntimeError(
                        f"Failed to generate embeddings for batch {i//batch_size + 1}: {e}. "
                        f"Please check your OpenAI API key and ensure it has access to embeddings."
                    )
        
        return all_embeddings


def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA256 hash of file contents.
    
    Args:
        file_path: Path to file
        
    Returns:
        Hexadecimal hash string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def compute_text_hash(text: str) -> str:
    """
    Compute SHA256 hash of text.
    
    Args:
        text: Text to hash
        
    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
