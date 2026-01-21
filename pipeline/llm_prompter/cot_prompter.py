"""Chain of Thought (CoT) Prompter - Step-by-step reasoning."""

import time
import logging
from typing import List, Optional

from config import Config
from .base import LLMPrompter


class ChainOfThoughtPrompter(LLMPrompter):
    """Chain of Thought prompting with step-by-step reasoning."""
    
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
        Initialize CoT Prompter.
        
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
            return "Baseline-CoT"
        elif self.prompt_mode == "improved":
            return "Improved-CoT"
        else:
            return "CoT"
    
    def _build_prompt(self, text: str, doc_id: Optional[str] = None) -> str:
        """Build the prompt for CoT prompting with improved reasoning steps."""
        if self.prompt_mode == "baseline":
            # Minimal baseline prompt
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
            # Improved baseline prompt - simplified 3-step reasoning + relation types
            prompt = f"""Extract biomedical relations from the following text using step-by-step reasoning.

Text:
{text}

REASONING STEPS:

Step 1: IDENTIFY ENTITIES
- Find all biomedical entities (genes, proteins, diseases, drugs, mutations, etc.)
- Use EXACT text spans from the document

Step 2: IDENTIFY RELATIONS
- For each pair of entities, determine if there is a relation
- Use the most complete entity description available

Step 3: VERIFY
- Check each entity mention appears exactly in the text
- Ensure relation types are correct
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
        
        # Full CoT prompt (default)
        prompt = self._build_base_prompt(text, doc_id)
        
        prompt += """TASK: Extract ALL biomedical relations from the text above using a step-by-step reasoning approach.

CRITICAL: Be THOROUGH and COMPREHENSIVE. Missing relations significantly hurts performance. Extract EVERYTHING.

REASONING STEPS:

Step 1: IDENTIFY ALL ENTITIES (Be Comprehensive)
- Scan the ENTIRE text MULTIPLE times for ALL biomedical entities
- Entities include: genes, proteins, diseases, drugs, mutations, variants, chemicals, organisms, etc.
- For each entity, extract the EXACT text span from the document
- Pay special attention to:
  * Complex entity mentions (mutations, variants, long descriptions)
  * Entities mentioned in parentheses, brackets, or complex phrases
  * Entities with abbreviations and full forms (use the MOST COMPLETE form)
  * Entities mentioned multiple times (use the most complete description)
- Think: "What are ALL the entities mentioned, including complex descriptions?"
- Count entities and ensure you haven't missed any

Step 2: CLASSIFY ENTITY TYPES (Optional but Helpful)
- Classify each identified entity's type (e.g., DiseaseOrPhenotypicFeature, GeneOrGeneProduct, OrganismTaxon, ChemicalEntity)
- This helps understand the context but is not required for the final output
- Helps identify potential relations between entities

Step 3: IDENTIFY ALL RELATIONS (Be Thorough - This is Critical)
- For each pair of entities, carefully determine if there is a relation
- Extract ALL relations, including:
  * Obvious relations (explicitly stated: "X causes Y", "X inhibits Y")
  * Implicit relations (strongly implied: "X is dependent on Y", "X requires Y")
  * Novel relations (newly discovered or unexpected)
  * Relations mentioned in different sentences or paragraphs
  * Relations involving multiple entities ("A, B, and C affect D" = extract 3 relations)
- For each relation, determine:
  * Head entity (use the exact text span from Step 1 - prefer longer, more complete forms)
  * Tail entity (use the exact text span from Step 1 - prefer longer, more complete forms)
  * Relation type (choose from the 8 allowed types - map carefully)
- When in doubt about whether a relation exists, INCLUDE IT (better to have it than miss it)
- Scan the text sentence by sentence to ensure nothing is missed

Step 4: VERIFY RELATIONS (Ensure Accuracy)
- Verify each relation is explicitly stated or strongly implied in the text
- Verify each entity mention appears EXACTLY in the source text (word-for-word match)
- Check that relation types match the semantic meaning in the text
- Use ONLY the 8 allowed relation types - map other concepts correctly
- Remove ONLY relations that are pure inference (not stated or implied)

Step 5: CHECK FOR MISSING RELATIONS (Final Pass - Critical)
- Review the ENTIRE text ONE MORE TIME to ensure you haven't missed any relations
- Pay special attention to:
  * Relations involving complex entity descriptions
  * Relations mentioned in passing or indirectly
  * Novel or unexpected relations that might be easy to miss
  * Relations in different parts of the text (title, abstract, different paragraphs)
- Count the number of relations and ensure you've been comprehensive
- Think: "Are there any relations I might have overlooked?"

"""
        
        # Add few-shot examples
        prompt += self._get_few_shot_examples()
        
        # Add relation type definitions
        prompt += self._get_relation_type_definitions()
        
        prompt += """
FINAL OUTPUT:
After completing all reasoning steps, provide your final answer as a JSON array.
Return ONLY the JSON array (no markdown code fences, no explanations after the JSON):

[
  {
    "head_mention": "exact text from document",
    "tail_mention": "exact text from document",
    "relation_type": "Association"
  }
]

CRITICAL OUTPUT RULES:
- Return ONLY the JSON array - no markdown code fences (```json), no explanations after
- Use ONLY the 8 ALLOWED relation types: Association, Positive_Correlation, Negative_Correlation, Bind, Cotreatment, Comparison, Drug_Interaction, Conversion
- DO NOT use types like "Causes", "Treats", "Regulates", "Part_Of", etc. - map them to allowed types
- Use EXACT text spans from the document for entity mentions (word-for-word match)
- Do not paraphrase or modify the text
- Extract ALL relations - be comprehensive. Missing relations hurts performance significantly.
- When unsure about type, use "Association"
- Before submitting, review to ensure you've extracted EVERYTHING - completeness is critical
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
