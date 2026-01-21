"""RAG (Retrieval-Augmented Generation) Prompter."""

import time
import logging
from typing import List, Optional

from config import Config
from .base import LLMPrompter
from ..retrieval import VectorStore


class RAGPrompter(LLMPrompter):
    """Retrieval-Augmented Generation prompting with external knowledge."""
    
    def __init__(
        self,
        entity_map=None,
        use_exact_spans: bool = True,
        include_relation_types: bool = True,
        model: Optional[str] = None,
        vector_store: Optional[VectorStore] = None,
        top_k: int = None,
        prompt_mode: str = "full",
        baseline_mode: bool = False,  # Deprecated, use prompt_mode instead
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize RAG Prompter.
        
        Args:
            entity_map: Optional global entity map
            use_exact_spans: Whether to encourage exact text span extraction
            include_relation_types: Whether to include relation type definitions in prompts
            model: Model name/key (defaults to config default)
            vector_store: Vector store instance (creates new one if None)
            top_k: Number of retrieved documents (defaults to config)
            prompt_mode: Prompt complexity level: "baseline", "improved", or "full" (default: "full")
            baseline_mode: Deprecated, use prompt_mode="baseline" instead
            logger: Optional logger instance
        """
        super().__init__(entity_map, use_exact_spans, include_relation_types, logger)
        # Handle backward compatibility with baseline_mode
        if baseline_mode:
            self.prompt_mode = "baseline"
        else:
            self.prompt_mode = prompt_mode
        self.model = Config.get_model_name(model)
        # Always use OpenRouter for all models (more stable tool calling)
        self.api_key = Config.OPENROUTER_API_KEY
        self.base_url = Config.OPENROUTER_BASE_URL
        self.model_name = self.model  # Use full model name with provider prefix
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found")
        if not self.base_url:
            raise ValueError("OPENROUTER_BASE_URL not set. Default should be 'https://openrouter.ai/api/v1'")
        self.logger.debug(f"[{self.name}] Using OpenRouter API with base_url: {self.base_url}")
        self.top_k = top_k or Config.RAG_TOP_K
        
        # Initialize or use provided vector store
        if vector_store is None:
            self.vector_store = VectorStore()
            # Load documents from source directory
            self.vector_store.add_documents_from_files(Config.RAG_SOURCE_DIR)
        else:
            self.vector_store = vector_store
    
    @property
    def name(self) -> str:
        """Return technique name."""
        if self.prompt_mode == "baseline":
            return "Baseline-RAG"
        elif self.prompt_mode == "improved":
            return "Improved-RAG"
        else:
            return "RAG"
    
    def _retrieve_context(self, text: str) -> str:
        """
        Retrieve relevant context from vector store.
        
        Args:
            text: Document text to use as query
            
        Returns:
            Retrieved context as formatted string
        """
        # Use first few sentences or a summary of the text as query
        # Try to extract key terms for better retrieval
        query = text[:1000] if len(text) > 1000 else text
        
        self.logger.debug(f"[{self.name}] Retrieving context with top_k={self.top_k}")
        results = self.vector_store.search(query, top_k=self.top_k)
        
        if not results:
            self.logger.debug(f"[{self.name}] No relevant context found")
            return "No relevant context found in the knowledge base."
        
        self.logger.debug(f"[{self.name}] Retrieved {len(results)} context documents")
        context_parts = []
        for i, result in enumerate(results, 1):
            similarity = result.get('similarity', 0.0)
            result_text = result.get('text', '')
            pmid = result.get('pmid', '')
            title = result.get('title', '')
            filename = result.get('filename', '')
            
            self.logger.debug(f"[{self.name}] Context {i}: similarity={similarity:.3f}, PMID={pmid or filename}")
            
            # Format context with metadata if available
            context_entry = f"[Context {i}]"
            if pmid:
                context_entry += f" PMID: {pmid}"
            elif filename:
                # Try to extract PMID from filename if it's a number
                context_entry += f" Source: {filename}"
            if title:
                context_entry += f"\nTitle: {title}"
            context_entry += f"\nSimilarity: {similarity:.3f}\n"
            
            # Include more text for better context (up to 800 chars)
            context_entry += f"{result_text[:800]}"
            if len(result_text) > 800:
                context_entry += "..."
            context_entry += "\n"
            
            context_parts.append(context_entry)
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, text: str, doc_id: Optional[str] = None) -> str:
        """Build the prompt for RAG prompting with improved instructions."""
        if self.prompt_mode == "baseline":
            # Minimal baseline prompt (skip retrieval)
            prompt = f"""Extract biomedical relations from the following text.

Text:
{text}

Extract relations in JSON format:

[
  {{
    "head_mention": "entity 1",
    "tail_mention": "entity 2",
    "relation_type": "relation type"
  }}
]
"""
            return prompt
        
        elif self.prompt_mode == "improved":
            # Improved baseline prompt - WITH retrieval + basic instructions + relation types
            context = self._retrieve_context(text)
            
            prompt = f"""Extract biomedical relations from the following text.

Text:
{text}

---

Relevant Context from PubMed:
{context}

---

TASK: Extract relations from the ORIGINAL text above (not from the context).

INSTRUCTIONS:
1. Use EXACT text spans from the original document for entity mentions
2. Use the context to understand entities and relationships
3. Only extract relations present in the original text
{self._get_brief_relation_type_definitions()}
{self._get_brief_few_shot_examples()}
OUTPUT FORMAT:
Return ONLY a valid JSON array (no markdown, no explanations):

[
  {{
    "head_mention": "exact text from document",
    "tail_mention": "exact text from document",
    "relation_type": "Association"
  }}
]
"""
            return prompt
        
        # Full RAG prompt with retrieval (default)
        context = self._retrieve_context(text)
        
        prompt = self._build_base_prompt(text, doc_id)
        
        prompt += f"""Relevant Context from PubMed Knowledge Base:
{context}

---

TASK: Extract ALL biomedical relations from the ORIGINAL document text above.

INSTRUCTIONS:
The context from PubMed abstracts provided above may help you:
1. Understand the biomedical entities mentioned in the document
2. Identify the correct relation types between entities
3. Verify entity names and their relationships
4. Understand complex entity descriptions and their variants
5. Identify novel relations that may be similar to known relations

However, you MUST:
- Extract entity mentions as EXACT text spans from the ORIGINAL document text (NOT from the context)
- Only extract relations that are present in the ORIGINAL document
- Use the context for understanding, but extract from the original text

CRITICAL RULES:
1. Extract ALL relations from the original document, including complex and novel ones
2. Use the most complete entity mention available in the ORIGINAL document
3. The context helps you understand what relations are possible, but extract only relations present in the original
4. Pay special attention to relations involving complex entity descriptions
5. Only extract relations that are explicitly stated in the original document

"""
        
        # Add few-shot examples
        prompt += self._get_few_shot_examples()
        
        # Add relation type definitions
        prompt += self._get_relation_type_definitions()
        
        prompt += """
OUTPUT FORMAT:
Return the results as a JSON array:
[
  {{
    "head_mention": "exact text from original document",
    "tail_mention": "exact text from original document",
    "relation_type": "Association"
  }}
]

REMEMBER: 
- Use EXACT text spans from the ORIGINAL document for entity mentions
- The context is only for understanding and verification - do NOT copy entity mentions from the context
- Extract ALL relations from the original document, even if they seem obvious
- The context helps you understand, but you must extract from the original text
"""
        return prompt
    
    def get_response(self, text: str, doc_id: Optional[str] = None) -> str:
        """
        Get LLM response using OpenRouter API with RAG.
        
        Args:
            text: Document text
            doc_id: Optional document ID
            
        Returns:
            LLM response string
        """
        self.logger.info(f"[{self.name}] Processing document: {doc_id}")
        self.logger.debug(f"[{self.name}] Document text length: {len(text)} characters")
        self.logger.debug(f"[{self.name}] Using model: {self.model_name} via OpenRouter")
        
        prompt = self._build_prompt(text, doc_id)
        self.logger.debug(f"[{self.name}] Prompt length: {len(prompt)} characters")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Responses API uses 'input' instead of 'messages'
        if Config.requires_responses_endpoint(self.model):
            payload = {
                "model": self.model_name,
                "input": prompt,  # Responses API uses 'input' instead of 'messages'
            }
            # Responses API uses 'max_output_tokens' instead of 'max_completion_tokens'
            # GPT-5 models need more tokens (reasoning tokens count as output)
            payload["max_output_tokens"] = Config.get_max_tokens_for_model(self.model)
            # GPT-5.x models only support default temperature (1.0), don't set temperature parameter
        else:
            # Chat Completions API uses 'messages'
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            }
            # GPT-5.x models only support default temperature (1.0), don't set temperature parameter
            if not Config.requires_default_temperature(self.model):
                payload["temperature"] = Config.TEMPERATURE
            # GPT-5.x and GPT-4.1 require max_completion_tokens instead of max_tokens
            # GPT-5 models need more tokens (reasoning tokens count as output)
            max_tokens = Config.get_max_tokens_for_model(self.model)
            if Config.requires_max_completion_tokens(self.model):
                payload["max_completion_tokens"] = max_tokens
            else:
                payload["max_tokens"] = max_tokens
        
        try:
            start_time = time.time()
            # GPT-5.x models use /responses endpoint, others use /chat/completions
            endpoint = "/responses" if Config.requires_responses_endpoint(self.model) else "/chat/completions"
            self.logger.info(f"[{self.name}] Sending request to OpenRouter API...")
            self.logger.debug(f"[{self.name}] Using endpoint: {endpoint}")
            
            response = self._make_api_request_with_retry(
                f"{self.base_url}{endpoint}",
                headers=headers,
                payload=payload,
                timeout=240  # Longer timeout for RAG (increased to 240 seconds)
            )
            result = response.json()
            
            # Responses API has different response structure
            if Config.requires_responses_endpoint(self.model):
                llm_response = self._extract_response_from_responses_api(result)
            else:
                # Chat Completions API structure
                llm_response = result["choices"][0]["message"]["content"]
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"[{self.name}] Received response in {elapsed_time:.2f} seconds")
            
            return llm_response
        except Exception as e:
            self.logger.error(f"[{self.name}] OpenRouter API error: {e}")
            raise
    
    def get_responses_batch(
        self, texts: List[str], doc_ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get responses for multiple documents (sequential for now).
        
        Args:
            texts: List of document texts
            doc_ids: Optional list of document IDs
            
        Returns:
            List of LLM responses
        """
        if doc_ids is None:
            doc_ids = [None] * len(texts)
        
        responses = []
        for text, doc_id in zip(texts, doc_ids):
            responses.append(self.get_response(text, doc_id))
        
        return responses
