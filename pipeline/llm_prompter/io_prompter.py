"""I/O (Input/Output) Prompter - Simple zero-shot prompting."""

import json
import time
import logging
from typing import List, Optional

from config import Config
from .base import LLMPrompter


class IOPrompter(LLMPrompter):
    """Simple zero-shot prompting for relation extraction."""
    
    def __init__(
        self,
        entity_map=None,
        use_exact_spans: bool = True,
        include_relation_types: bool = True,
        model: Optional[str] = None,
        prompt_mode: str = "full",
        baseline_mode: bool = False,  # Deprecated, use prompt_mode instead
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize IO Prompter.
        
        Args:
            entity_map: Optional global entity map
            use_exact_spans: Whether to encourage exact text span extraction
            include_relation_types: Whether to include relation type definitions in prompts
            model: Model name/key (defaults to config default)
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
    
    @property
    def name(self) -> str:
        """Return technique name."""
        if self.prompt_mode == "baseline":
            return "Baseline-IO"
        elif self.prompt_mode == "improved":
            return "Improved-IO"
        else:
            return "IO"
    
    def _build_prompt(self, text: str, doc_id: Optional[str] = None) -> str:
        """Build the prompt for IO prompting with few-shot examples."""
        if self.prompt_mode == "baseline":
            # Minimal baseline prompt with relation types list
            prompt = f"""Extract biomedical relations from the following text.

Text:
{text}

Relation types: Association, Positive_Correlation, Negative_Correlation, Bind, Cotreatment, Comparison, Drug_Interaction, Conversion

Extract relations in JSON format:

[
  {{
    "head_mention": "entity 1",
    "tail_mention": "entity 2",
    "relation_type": "Association"
  }}
]
"""
            return prompt
        
        elif self.prompt_mode == "improved":
            # Improved baseline prompt - adds relation types, basic examples, and entity rules
            prompt = f"""Extract biomedical relations from the following text.

Text:
{text}

TASK: Extract biomedical relations from the text above.

INSTRUCTIONS:
1. Use EXACT text spans from the document for entity mentions
2. Use the MOST COMPLETE entity description available
3. Extract relations that are explicitly stated in the text
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
        
        # Full prompt (default)
        prompt = self._build_base_prompt(text, doc_id)
        
        prompt += """TASK: Extract ALL biomedical relations from the text above.

CRITICAL INSTRUCTIONS FOR MAXIMUM COMPLETENESS:
1. Extract ALL relations - be thorough and comprehensive. Missing relations significantly hurts performance.
2. Include relations involving complex entity descriptions (mutations, variants, long descriptive phrases)
3. Extract BOTH explicitly stated AND strongly implied relations (e.g., "X is dependent on Y" implies a relation)
4. Use the MOST COMPLETE entity mention available in the text (prefer longer, more descriptive forms)
5. Extract relations even if they seem obvious or redundant
6. Pay special attention to:
   - Relations mentioned in different parts of the text (scan the ENTIRE text multiple times)
   - Novel or newly discovered relations
   - Relations with complex entity descriptions (mutations, variants, deletions, etc.)
   - Relations involving multiple entities (e.g., "A, B, and C are related to D" = extract 3 relations)
7. Only extract relations that are EXPLICITLY stated or STRONGLY IMPLIED in the text (avoid pure inference)
8. Before finalizing, verify each entity mention appears exactly in the text (use exact text spans)

EXTRACTION STRATEGY:
- Read through the entire text once to identify all entities
- Read through again to identify all relations between entities
- For each sentence, ask: "What relations are mentioned here?"
- Don't skip relations just because they seem obvious or already extracted
- When in doubt about whether a relation exists, extract it (better to have it than miss it)

"""
        
        # Add few-shot examples
        prompt += self._get_few_shot_examples()
        
        # Add relation type definitions
        prompt += self._get_relation_type_definitions()
        
        prompt += """
OUTPUT FORMAT:
Return ONLY a valid JSON array (no markdown, no code fences, no explanations). The JSON must be parseable.

[
  {
    "head_mention": "exact text from document",
    "tail_mention": "exact text from document",
    "relation_type": "Association"
  }
]

CRITICAL OUTPUT RULES:
- Return ONLY the JSON array - no markdown code fences (```json), no explanations before or after
- Use ONLY the 8 ALLOWED relation types: Association, Positive_Correlation, Negative_Correlation, Bind, Cotreatment, Comparison, Drug_Interaction, Conversion
- DO NOT use types like "Causes", "Treats", "Regulates", "Part_Of", etc. - map them to allowed types
- Use EXACT text spans from the document for entity mentions (word-for-word match)
- Do not paraphrase or modify the text
- Extract ALL relations - be comprehensive. Missing relations hurts performance significantly.
- Pay special attention to relations involving mutations, variants, and complex descriptions
- When unsure about type, use "Association" - it's the safest choice
- Review your output to ensure you've extracted EVERYTHING - completeness is critical
"""
        return prompt
    
    def get_response(self, text: str, doc_id: Optional[str] = None) -> str:
        """
        Get LLM response using OpenRouter API.
        
        Args:
            text: Document text
            doc_id: Optional document ID
            
        Returns:
            LLM response string (extracted content)
            
        """
        self.logger.info(f"[{self.name}] Processing document: {doc_id}")
        self.logger.debug(f"[{self.name}] Document text length: {len(text)} characters")
        self.logger.info(f"[{self.name}] Using model: {self.model_name} via OpenRouter")
        self.logger.debug(f"[{self.name}] Full model identifier: {self.model}")
        
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
                timeout=180  # Increased timeout to 180 seconds
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
            
            # Check for empty or suspiciously short responses
            if not llm_response or not llm_response.strip():
                self.logger.warning(f"[{self.name}] WARNING: Received empty or whitespace-only response!")
                self.logger.debug(f"[{self.name}] Full API response was: {json.dumps(result, indent=2)[:1000]}")
            elif len(llm_response.strip()) < 50:
                self.logger.warning(f"[{self.name}] WARNING: Response is suspiciously short ({len(llm_response)} chars). Extracted: {repr(llm_response[:100])}")
            
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
