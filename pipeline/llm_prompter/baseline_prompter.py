"""Baseline Prompter - Absolute minimal prompting for relation extraction."""

import json
import time
import logging
from typing import List, Optional

from config import Config
from .base import LLMPrompter
from ..prompts import get_relation_type_definitions


class BaselinePrompter(LLMPrompter):
    """Absolute minimal baseline prompting - just text and output format."""
    
    def __init__(
        self,
        entity_map=None,
        use_exact_spans: bool = True,
        include_relation_types: bool = False,  # Don't include relation types for baseline
        model: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize Baseline Prompter.
        
        Args:
            entity_map: Optional global entity map (not used in baseline)
            use_exact_spans: Whether to encourage exact text span extraction
            include_relation_types: Whether to include relation type definitions (default: False for baseline)
            model: Model name/key (defaults to config default)
            logger: Optional logger instance
        """
        super().__init__(entity_map, use_exact_spans, include_relation_types, logger)
        self.model = Config.get_model_name(model)
        # Always use OpenRouter for all models (more stable tool calling)
        self.api_key = Config.OPENROUTER_API_KEY
        self.base_url = Config.OPENROUTER_BASE_URL
        self.model_name = self.model  # Use full model name with provider prefix
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found")
        if not self.base_url:
            raise ValueError("OPENROUTER_BASE_URL not set. Default should be 'https://openrouter.ai/api/v1'")
        self.logger.debug(f"[{self.name}] Using OpenRouter API")
    
    @property
    def name(self) -> str:
        """Return technique name."""
        return "Baseline"
    
    def _build_prompt(self, text: str, doc_id: Optional[str] = None) -> str:
        """Build the baseline prompt with relation type definitions."""
        relation_types = get_relation_type_definitions()

        prompt = f"""Extract biomedical relations from the following text.
{relation_types}
Text:
{text}

Extract relations in JSON format:

[
  {{
    "head_mention": "entity 1",
    "tail_mention": "entity 2",
    "relation_type": "one of the 8 allowed relation types"
  }}
]

IMPORTANT: Use ONLY the 8 allowed relation types listed above.
"""
        return prompt
    
    def get_response(self, text: str, doc_id: Optional[str] = None) -> str:
        """
        Get LLM response using OpenRouter API.
        
        Args:
            text: Document text
            doc_id: Optional document ID
            
        Returns:
            LLM response string
        """
        self.logger.info(f"[{self.name}] Processing document: {doc_id}")
        
        prompt = self._build_prompt(text, doc_id)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Always use OpenRouter format
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": Config.TEMPERATURE,
            "max_tokens": Config.MAX_TOKENS,
        }
        url = f"{self.base_url}/chat/completions"

        response = self._make_api_request_with_retry(url, headers, payload)
        result = response.json()

        # Extract response content (OpenRouter format)
        if "choices" in result and len(result["choices"]) > 0:
            llm_response = result["choices"][0]["message"]["content"]
        else:
            self.logger.error(f"[{self.name}] Unexpected OpenRouter response structure: {result}")
            llm_response = ""

        return llm_response
    
    def get_responses_batch(
        self, texts: List[str], doc_ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get LLM responses for multiple documents sequentially.
        
        Args:
            texts: List of document texts
            doc_ids: Optional list of document IDs
            
        Returns:
            List of LLM responses
        """
        responses = []
        for i, text in enumerate(texts):
            doc_id = doc_ids[i] if doc_ids and i < len(doc_ids) else None
            response = self.get_response(text, doc_id)
            responses.append(response)
            # Small delay between requests to avoid rate limits
            if i < len(texts) - 1:
                time.sleep(0.5)
        return responses

