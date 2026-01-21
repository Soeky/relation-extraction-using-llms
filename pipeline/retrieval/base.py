"""Base retriever interface for RAG."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class Retriever(ABC):
    """Abstract base class for retrieval components."""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents/context for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents with metadata
        """
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the retrieval index.
        
        Args:
            documents: List of documents with text and metadata
        """
        pass
