"""Vector store with file-based embedding caching."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from config import Config
from .embeddings import EmbeddingGenerator, compute_file_hash, compute_text_hash


class VectorStore:
    """Vector store with file-based embedding caching and hash checking."""
    
    def __init__(self, embeddings_dir: Path = None):
        """
        Initialize vector store.
        
        Args:
            embeddings_dir: Directory to store embeddings cache
        """
        self.embeddings_dir = embeddings_dir or Config.RAG_EMBEDDINGS_DIR
        self.embeddings_dir.mkdir(exist_ok=True)
        
        self.embedding_generator = EmbeddingGenerator()
        self.embeddings: List[List[float]] = []
        self.documents: List[Dict[str, Any]] = []
        self.hash_index_path = self.embeddings_dir / "hash_index.json"
        self.embeddings_file = self.embeddings_dir / "embeddings.npy"
        self.documents_file = self.embeddings_dir / "documents.json"
        
        # Load existing index if available
        self.hash_index: Dict[str, str] = self._load_hash_index()
        self._load_cached_embeddings()
    
    def _load_hash_index(self) -> Dict[str, str]:
        """Load hash index mapping file paths to hashes."""
        if self.hash_index_path.exists():
            try:
                with open(self.hash_index_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_hash_index(self) -> None:
        """Save hash index to disk."""
        with open(self.hash_index_path, 'w') as f:
            json.dump(self.hash_index, f, indent=2)
    
    def _load_cached_embeddings(self) -> None:
        """Load cached embeddings and documents from disk."""
        if self.embeddings_file.exists() and self.documents_file.exists():
            try:
                self.embeddings = np.load(self.embeddings_file).tolist()
                with open(self.documents_file, 'r') as f:
                    self.documents = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load cached embeddings: {e}")
                self.embeddings = []
                self.documents = []
    
    def _save_embeddings(self) -> None:
        """Save embeddings and documents to disk."""
        np.save(self.embeddings_file, np.array(self.embeddings))
        with open(self.documents_file, 'w') as f:
            json.dump(self.documents, f, indent=2)
    
    def add_documents_from_files(self, source_dir: Path) -> None:
        """
        Add documents from source directory with hash checking.
        
        Args:
            source_dir: Directory containing source files
        """
        if not source_dir.exists():
            print(f"Warning: Source directory {source_dir} does not exist")
            return
        
        new_documents = []
        files_to_process = []
        embedding_start_idx = len(self.embeddings)
        
        # Check all files in source directory
        for file_path in sorted(source_dir.rglob("*")):
            if file_path.is_file() and file_path.suffix in ['.txt', '.md', '.json']:
                current_hash = compute_file_hash(file_path)
                stored_hash = self.hash_index.get(str(file_path))
                
                if stored_hash != current_hash:
                    # File changed or new file
                    files_to_process.append((file_path, current_hash))
                else:
                    # File unchanged, find existing document
                    existing_doc = next(
                        (doc for doc in self.documents 
                         if doc.get('file_path') == str(file_path)),
                        None
                    )
                    if existing_doc is not None:
                        # Preserve embedding index
                        new_documents.append(existing_doc.copy())
        
        # Process changed/new files
        if files_to_process:
            print(f"Processing {len(files_to_process)} new/changed files...")
            texts = []
            new_docs_to_add = []
            for file_path, file_hash in files_to_process:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    doc = {
                        'file_path': str(file_path),
                        'text': text,
                        'hash': file_hash,
                        'filename': file_path.name
                    }
                    texts.append(text)
                    new_docs_to_add.append(doc)
                    self.hash_index[str(file_path)] = file_hash
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
            
            # Generate embeddings for new/changed files
            if texts:
                print(f"Generating embeddings for {len(texts)} documents...")
                new_embeddings = self.embedding_generator.generate_embeddings_batch(texts)
                
                # Add to existing embeddings and set indices
                for doc, embedding in zip(new_docs_to_add, new_embeddings):
                    idx = len(self.embeddings)
                    self.embeddings.append(embedding)
                    doc['embedding_index'] = idx
                    new_documents.append(doc)
        else:
            print("No new or changed files detected. Using cached embeddings.")
        
        # Update documents list
        self.documents = new_documents
        
        # Save updated index and embeddings
        self._save_hash_index()
        self._save_embeddings()
        print(f"Vector store now contains {len(self.documents)} documents")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents directly (for programmatic addition).
        
        Args:
            documents: List of documents with 'text' field
        """
        texts = [doc['text'] for doc in documents]
        
        # Check for existing documents by text hash
        new_texts = []
        new_docs = []
        for doc, text in zip(documents, texts):
            text_hash = compute_text_hash(text)
            existing_idx = next(
                (i for i, d in enumerate(self.documents) 
                 if d.get('text_hash') == text_hash),
                None
            )
            
            if existing_idx is None:
                doc['text_hash'] = text_hash
                new_texts.append(text)
                new_docs.append(doc)
            else:
                # Reuse existing embedding
                doc['embedding_index'] = self.documents[existing_idx]['embedding_index']
                new_docs.append(doc)
        
        # Generate embeddings for new documents
        if new_texts:
            new_embeddings = self.embedding_generator.generate_embeddings_batch(new_texts)
            for doc, embedding in zip(new_docs[-len(new_texts):], new_embeddings):
                idx = len(self.embeddings)
                self.embeddings.append(embedding)
                doc['embedding_index'] = idx
        
        self.documents.extend(new_docs)
        self._save_embeddings()
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of similar documents with similarity scores
        """
        if not self.embeddings or not self.documents:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Compute cosine similarity
        embeddings_array = np.array(self.embeddings)
        query_array = np.array(query_embedding)
        
        # Normalize for cosine similarity
        query_norm = query_array / (np.linalg.norm(query_array) + 1e-10)
        embeddings_norm = embeddings_array / (np.linalg.norm(embeddings_array, axis=1, keepdims=True) + 1e-10)
        
        similarities = np.dot(embeddings_norm, query_norm)
        
        # Get top_k indices (these are embedding array indices)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return documents with similarity scores
        # Find documents that match these embedding indices
        results = []
        for emb_idx in top_indices:
            # Find document(s) with this embedding index
            matching_docs = [
                doc for doc in self.documents 
                if doc.get('embedding_index') == emb_idx
            ]
            for doc in matching_docs:
                doc_copy = doc.copy()
                doc_copy['similarity'] = float(similarities[emb_idx])
                results.append(doc_copy)
        
        # Remove duplicates and limit to top_k
        seen = set()
        unique_results = []
        for doc in results:
            doc_id = doc.get('file_path') or doc.get('text_hash') or id(doc)
            if doc_id not in seen:
                seen.add(doc_id)
                unique_results.append(doc)
                if len(unique_results) >= top_k:
                    break
        
        return unique_results
    
    def clear(self) -> None:
        """Clear all embeddings and documents."""
        self.embeddings = []
        self.documents = []
        self.hash_index = {}
        if self.embeddings_file.exists():
            self.embeddings_file.unlink()
        if self.documents_file.exists():
            self.documents_file.unlink()
        if self.hash_index_path.exists():
            self.hash_index_path.unlink()
