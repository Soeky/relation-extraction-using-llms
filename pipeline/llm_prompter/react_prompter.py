import time
import logging
from typing import List, Optional

from config import Config
from .base import LLMPrompter


class ReActPrompter(LLMPrompter):

    def __init__(
        self,
        entity_map=None,
        use_exact_spans: bool = True,
        include_relation_types: bool = True,
        model: Optional[str] = None,
        prompt_mode: str = "full",
        baseline_mode: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(entity_map, use_exact_spans, include_relation_types, logger)
        if baseline_mode:
            self.prompt_mode = "baseline"
        else:
            self.prompt_mode = prompt_mode
        self.model = Config.get_model_name(model)
        self.api_key = Config.OPENROUTER_API_KEY
        self.base_url = Config.OPENROUTER_BASE_URL
        self.model_name = self.model
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found")
        if not self.base_url:
            raise ValueError("OPENROUTER_BASE_URL not set. Default should be 'https://openrouter.ai/api/v1'")
        self.logger.debug(f"[{self.name}] Using OpenRouter API with base_url: {self.base_url}")

    @property
    def name(self) -> str:
        if self.prompt_mode == "baseline":
            return "Baseline-ReAct"
        elif self.prompt_mode == "improved":
            return "Improved-ReAct"
        else:
            return "ReAct"

    def _build_prompt(self, text: str, doc_id: Optional[str] = None) -> str:
        if self.prompt_mode == "baseline":
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
            prompt = f"""Extract biomedical relations from the following text using structured reasoning.

Text:
{text}

Use the following OBSERVE-THINK-ACT-REFLECT-EXTRACT workflow:

OBSERVE: Identify all biomedical entities in the text
- Genes, proteins, diseases, drugs, chemicals, mutations, variants
- Use EXACT text spans from the document

THINK: Consider relationships between identified entities
- What interactions are mentioned or implied?
- What evidence supports each relationship?

ACT: Determine the relation type for each pair
- Map to one of the 8 allowed types

REFLECT: Verify your findings
- Is each entity mention exact from the text?
- Is the relation type accurate?

EXTRACT: Provide the final JSON output
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

        prompt = self._build_base_prompt(text, doc_id)

        prompt += """TASK: Extract ALL biomedical relations from the text above using the ReAct reasoning framework.

CRITICAL: Be THOROUGH and COMPREHENSIVE. Missing relations significantly hurts performance. Extract EVERYTHING.

Use the OBSERVE-THINK-ACT-REFLECT-EXTRACT workflow:

OBSERVE: Identify All Biomedical Entities
- Scan the ENTIRE text MULTIPLE times for ALL biomedical entities
- Entities include: genes, proteins, diseases, drugs, mutations, variants, chemicals, organisms
- For each entity, extract the EXACT text span from the document
- Pay special attention to:
  * Complex entity mentions (mutations, variants, long descriptions)
  * Entities mentioned in parentheses, brackets, or complex phrases
  * Entities with abbreviations and full forms (use the MOST COMPLETE form)
  * Entities mentioned multiple times (use the most complete description)
- Think: "What are ALL the entities mentioned, including complex descriptions?"
- Count entities and ensure you haven't missed any

THINK: Consider Relationships Between Entities
- For each pair of entities, carefully consider if there is a relation
- Consider ALL types of relations:
  * Obvious relations (explicitly stated: "X causes Y", "X inhibits Y")
  * Implicit relations (strongly implied: "X is dependent on Y", "X requires Y")
  * Novel relations (newly discovered or unexpected)
  * Relations mentioned in different sentences or paragraphs
  * Relations involving multiple entities ("A, B, and C affect D" = 3 relations)
- Think about the evidence for each relationship
- When in doubt about whether a relation exists, INCLUDE IT (better to have it than miss it)

ACT: Determine Relation Types
- For each identified relation, determine:
  * Head entity (use the exact text span - prefer longer, more complete forms)
  * Tail entity (use the exact text span - prefer longer, more complete forms)
  * Relation type (choose from the 8 allowed types - map carefully)
- Use ONLY the 8 allowed relation types - map other concepts correctly
- When unsure about type, use "Association" - it's the safest choice

REFLECT: Verify Your Findings
- Verify each relation is explicitly stated or strongly implied in the text
- Verify each entity mention appears EXACTLY in the source text (word-for-word match)
- Check that relation types match the semantic meaning in the text
- Remove ONLY relations that are pure inference (not stated or implied)
- Review the ENTIRE text ONE MORE TIME to ensure you haven't missed any relations
- Pay special attention to relations in different parts of the text

EXTRACT: Provide Final JSON Output
- Compile all verified relations into the final JSON array
- Ensure completeness - count relations and verify you've been thorough

"""

        prompt += self._get_few_shot_examples()

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
        self.logger.info(f"[{self.name}] Processing document: {doc_id}")
        self.logger.debug(f"[{self.name}] Document text length: {len(text)} characters")
        self.logger.debug(f"[{self.name}] Using model: {self.model_name} via OpenRouter")

        prompt = self._build_prompt(text, doc_id)
        self.logger.debug(f"[{self.name}] Prompt length: {len(prompt)} characters")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if Config.requires_responses_endpoint(self.model):
            payload = {
                "model": self.model_name,
                "input": prompt,
            }
            payload["max_output_tokens"] = Config.get_max_tokens_for_model(self.model)
        else:
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            }

            if not Config.requires_default_temperature(self.model):
                payload["temperature"] = Config.TEMPERATURE

            max_tokens = Config.get_max_tokens_for_model(self.model)
            if Config.requires_max_completion_tokens(self.model):
                payload["max_completion_tokens"] = max_tokens
            else:
                payload["max_tokens"] = max_tokens

        try:
            start_time = time.time()

            endpoint = "/responses" if Config.requires_responses_endpoint(self.model) else "/chat/completions"
            self.logger.info(f"[{self.name}] Sending request to OpenRouter API...")
            self.logger.debug(f"[{self.name}] Using endpoint: {endpoint}")

            response = self._make_api_request_with_retry(
                f"{self.base_url}{endpoint}",
                headers=headers,
                payload=payload,
                timeout=180
            )
            result = response.json()

            if Config.requires_responses_endpoint(self.model):
                llm_response = self._extract_response_from_responses_api(result)
            else:
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
        if doc_ids is None:
            doc_ids = [None] * len(texts)

        responses = []
        for text, doc_id in zip(texts, doc_ids):
            responses.append(self.get_response(text, doc_id))

        return responses
