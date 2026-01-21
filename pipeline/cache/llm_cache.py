"""LLM response cache for reusing responses across runs."""

import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class LLMResponseCache:
    """Cache for LLM responses to avoid re-querying the same prompts."""
    
    def __init__(
        self,
        cache_dir: Path,
        enabled: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize LLM response cache.
        
        Args:
            cache_dir: Directory to store cached responses
            enabled: Whether caching is enabled
            logger: Optional logger instance
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.logger = logger or logging.getLogger(__name__)
        
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"[LLMCache] Cache enabled, directory: {self.cache_dir}")
        else:
            self.logger.info("[LLMCache] Cache disabled")
    
    def _extract_prompt_template(self, prompt: str) -> str:
        """
        Extract the prompt template (without document text) from a full prompt.
        
        Args:
            prompt: Full prompt text
            
        Returns:
            Prompt template string (prompt without document text)
        """
        # Try to find where the document text starts
        text_markers = ["Text:\n", "Document Text:\n", "TEXT:\n"]
        text_start = -1
        
        for marker in text_markers:
            pos = prompt.find(marker)
            if pos != -1:
                text_start = pos + len(marker)
                break
        
        if text_start == -1:
            # No document text marker found, return full prompt as template
            return prompt
        
        # Find where the document text ends (look for common separators)
        text_end = prompt.find("\n---", text_start)
        if text_end == -1:
            text_end = prompt.find("\nRelevant Context", text_start)
        if text_end == -1:
            text_end = prompt.find("\nTASK:", text_start)
        if text_end == -1:
            text_end = prompt.find("\nINSTRUCTIONS:", text_start)
        if text_end == -1:
            # If no separator found, assume rest is document text
            # Return everything before text_start as template
            return prompt[:text_start]
        
        # Extract template: everything before document text + everything after document text
        template = prompt[:text_start] + prompt[text_end:]
        return template
    
    def _get_cache_key(
        self,
        technique: str,
        model: str,
        doc_id: str,
        prompt: str
    ) -> str:
        """
        Generate cache key from technique, model, doc_id, prompt template, and full prompt hash.
        
        The cache key structure ensures that:
        - Different techniques are cached separately
        - Different models are cached separately  
        - Different prompt templates are cached separately
        - Different documents are cached separately
        
        Args:
            technique: Prompting technique name (e.g., "IO", "CoT")
            model: Model name
            doc_id: Document ID
            prompt: Prompt text
            
        Returns:
            Cache key string with structure: technique/model/template_hash/doc_id_docHash_promptHash.json
        """
        # Extract prompt template (without document text) and hash it
        # This ensures different prompt templates are cached separately
        prompt_template = self._extract_prompt_template(prompt)
        template_hash = hashlib.sha256(prompt_template.encode('utf-8')).hexdigest()[:12]
        
        # Create hash of full prompt to detect any changes
        prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:12]
        
        # Include document text hash in filename for easier lookup by document
        # This allows us to find cached responses by document text without building the prompt
        doc_hash = ""
        if "Text:\n" in prompt:
            try:
                text_start = prompt.find("Text:\n") + len("Text:\n")
                # Look for common separators that come after the text
                text_end = prompt.find("\n---", text_start)
                if text_end == -1:
                    text_end = prompt.find("\nRelevant Context", text_start)
                if text_end == -1:
                    text_end = prompt.find("\nTASK:", text_start)
                if text_end == -1:
                    text_end = prompt.find("\nINSTRUCTIONS:", text_start)
                if text_end == -1:
                    # If no separator found, take a reasonable chunk (first 2000 chars)
                    text_end = min(text_start + 2000, len(prompt))
                
                if text_end > text_start:
                    doc_text = prompt[text_start:text_end].strip()
                    doc_hash = hashlib.sha256(doc_text.encode('utf-8')).hexdigest()[:12] + "_"
            except Exception:
                pass
        
        # Sanitize model name for filesystem
        model_safe = model.replace('/', '_').replace('-', '_')
        
        # Cache key format: technique/model/template_hash/doc_id_[dochash_]prompthash.json
        # This structure makes it clear which prompt template version is being used
        return f"{technique}/{model_safe}/{template_hash}/{doc_id}_{doc_hash}{prompt_hash}.json"
    
    def _get_document_cache_key(
        self,
        technique: str,
        model: str,
        doc_id: str,
        document_text: str
    ) -> str:
        """
        Generate cache key from technique, model, doc_id, and document text hash.
        This is used to check cache BEFORE building the prompt (useful for RAG).
        
        Args:
            technique: Prompting technique name
            model: Model name
            doc_id: Document ID
            document_text: Document text (before prompt building)
            
        Returns:
            Cache key string (wildcard pattern to match any prompt hash)
        """
        # Create hash of document text
        doc_hash = hashlib.sha256(document_text.encode('utf-8')).hexdigest()[:16]
        
        # Sanitize model name for filesystem
        model_safe = model.replace('/', '_').replace('-', '_')
        
        # Return pattern: technique/model/doc_id_docHash_*.json
        # We'll search for files matching this pattern
        return f"{technique}/{model_safe}/{doc_id}_{doc_hash}_*.json"
    
    def get_by_document(
        self,
        technique: str,
        model: str,
        doc_id: str,
        document_text: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached response by document text (before prompt building).
        This is useful for RAG where we want to avoid retrieval if cached.
        
        Args:
            technique: Prompting technique name
            model: Model name
            doc_id: Document ID
            document_text: Document text
            
        Returns:
            Cached response dict, or None if not found
        """
        if not self.enabled:
            return None
        
        # Create document hash
        doc_hash = hashlib.sha256(document_text.encode('utf-8')).hexdigest()[:12]
        
        # Sanitize model name
        model_safe = model.replace('/', '_').replace('-', '_')
        
        # Search for cache files matching this document
        # Need to search in all template_hash subdirectories
        cache_base_dir = self.cache_dir / technique / model_safe
        if not cache_base_dir.exists():
            return None
        
        # Look for files starting with doc_id_docHash_ in any template_hash subdirectory
        pattern_prefix = f"{doc_id}_{doc_hash}_"
        
        for template_dir in cache_base_dir.iterdir():
            if not template_dir.is_dir():
                continue
            for cache_file in template_dir.glob(f"{pattern_prefix}*.json"):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    
                    # Verify it's for the same document text
                    cached_doc_text = cached_data.get("metadata", {}).get("document_text_hash")
                    if cached_doc_text and cached_doc_text[:12] == doc_hash:
                        self.logger.debug(
                            f"[LLMCache] Cache HIT (by document): {technique}/{model}/{doc_id} "
                            f"(template: {template_dir.name}, cached: {cached_data.get('metadata', {}).get('cached_at', 'unknown')})"
                        )
                        return cached_data
                except Exception as e:
                    self.logger.warning(
                        f"[LLMCache] Error reading cache file {cache_file}: {e}"
                    )
                    continue
        
        self.logger.debug(f"[LLMCache] Cache MISS (by document): {technique}/{model}/{doc_id}")
        return None
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get full path to cache file."""
        return self.cache_dir / cache_key
    
    def get(
        self,
        technique: str,
        model: str,
        doc_id: str,
        prompt: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached response if available.
        
        Args:
            technique: Prompting technique name
            model: Model name
            doc_id: Document ID
            prompt: Prompt text
            
        Returns:
            Cached response dict with 'response' and 'metadata' keys, or None if not found
        """
        if not self.enabled:
            return None
        
        cache_key = self._get_cache_key(technique, model, doc_id, prompt)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                self.logger.debug(
                    f"[LLMCache] Cache HIT: {technique}/{model}/{doc_id} "
                    f"(cached: {cached_data.get('metadata', {}).get('cached_at', 'unknown')})"
                )
                return cached_data
            except Exception as e:
                self.logger.warning(
                    f"[LLMCache] Error reading cache file {cache_path}: {e}"
                )
                return None
        
        self.logger.debug(f"[LLMCache] Cache MISS: {technique}/{model}/{doc_id}")
        return None
    
    def save(
        self,
        technique: str,
        model: str,
        doc_id: str,
        prompt: str,
        response: str,
        document_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save response to cache.
        
        Args:
            technique: Prompting technique name
            model: Model name
            doc_id: Document ID
            prompt: Prompt text
            response: LLM response text
            metadata: Optional metadata to store with response
            
        Returns:
            Path to saved cache file
        """
        if not self.enabled:
            return Path()
        
        cache_key = self._get_cache_key(technique, model, doc_id, prompt)
        cache_path = self._get_cache_path(cache_key)
        
        # Create parent directory if needed
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract prompt template hash for metadata
        prompt_template = self._extract_prompt_template(prompt)
        template_hash = hashlib.sha256(prompt_template.encode('utf-8')).hexdigest()[:12]
        
        # Extract document text hash for document-based cache lookup
        document_text_hash = None
        if document_text:
            # Use provided document text directly
            document_text_hash = hashlib.sha256(document_text.encode('utf-8')).hexdigest()[:12]
        elif "Text:\n" in prompt:
            # Try to extract the original document text from the prompt
            try:
                text_start = prompt.find("Text:\n") + len("Text:\n")
                # Look for common separators that come after the text
                text_end = prompt.find("\n---", text_start)
                if text_end == -1:
                    text_end = prompt.find("\nRelevant Context", text_start)
                if text_end == -1:
                    text_end = prompt.find("\nTASK:", text_start)
                if text_end == -1:
                    text_end = prompt.find("\nINSTRUCTIONS:", text_start)
                if text_end == -1:
                    # If no separator found, take a reasonable chunk (first 2000 chars)
                    text_end = min(text_start + 2000, len(prompt))
                
                if text_end > text_start:
                    doc_text = prompt[text_start:text_end].strip()
                    document_text_hash = hashlib.sha256(doc_text.encode('utf-8')).hexdigest()[:12]
            except Exception:
                pass
        
        # Prepare cache data
        cache_data = {
            "technique": technique,
            "model": model,
            "doc_id": doc_id,
            "prompt": prompt,
            "response": response,
            "metadata": {
                "cached_at": datetime.now().isoformat(),
                "prompt_template_hash": template_hash,
                "prompt_hash": hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:12],
                "document_text_hash": document_text_hash,
                **(metadata or {})
            }
        }
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"[LLMCache] Saved to cache: {cache_path}")
            return cache_path
        except Exception as e:
            self.logger.warning(f"[LLMCache] Error saving to cache {cache_path}: {e}")
            return Path()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache stats (total files, size, etc.)
        """
        if not self.enabled or not self.cache_dir.exists():
            return {"enabled": False, "total_files": 0, "total_size_bytes": 0}
        
        total_files = 0
        total_size = 0
        
        for cache_file in self.cache_dir.rglob("*.json"):
            if cache_file.is_file():
                total_files += 1
                total_size += cache_file.stat().st_size
        
        return {
            "enabled": True,
            "cache_dir": str(self.cache_dir),
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }

