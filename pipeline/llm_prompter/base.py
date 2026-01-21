"""Base class for LLM prompters."""

import logging
import re
import time
from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING
import requests

if TYPE_CHECKING:
    from ..data.entity_map import GlobalEntityMap


class LLMPrompter(ABC):
    """Abstract base class for LLM prompting techniques."""
    
    def __init__(
        self,
        entity_map: Optional["GlobalEntityMap"] = None,
        use_exact_spans: bool = True,
        include_relation_types: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the prompter.
        
        Args:
            entity_map: Optional global entity map for context
            use_exact_spans: Whether to encourage exact text span extraction
            include_relation_types: Whether to include relation type definitions in prompts
            logger: Optional logger instance
        """
        self.entity_map = entity_map
        self.use_exact_spans = use_exact_spans
        self.include_relation_types = include_relation_types
        self.logger = logger or logging.getLogger(__name__)
    
    @abstractmethod
    def get_response(self, text: str, doc_id: Optional[str] = None) -> str:
        """
        Get LLM response for a single document text.
        
        Args:
            text: Document text (title + body)
            doc_id: Optional document ID for context
            
        Returns:
            LLM response as string
        """
        pass
    
    @abstractmethod
    def get_responses_batch(
        self, texts: List[str], doc_ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get LLM responses for multiple documents (optional optimization).
        
        Args:
            texts: List of document texts
            doc_ids: Optional list of document IDs
            
        Returns:
            List of LLM responses
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this prompting technique."""
        pass
    
    def _build_base_prompt(self, text: str, doc_id: Optional[str] = None) -> str:
        """
        Build base prompt with common instructions.
        
        Args:
            text: Document text
            doc_id: Optional document ID
            
        Returns:
            Base prompt string
        """
        prompt = """Extract biomedical relations from the following text.

"""
        if doc_id:
            prompt += f"Document ID: {doc_id}\n\n"
        
        if self.use_exact_spans:
            prompt += """CRITICAL INSTRUCTIONS FOR ENTITY EXTRACTION:
1. Use the EXACT text spans from the document for all entity mentions
2. Do NOT paraphrase, summarize, or modify entity mentions
3. Copy entity mentions exactly as they appear in the text, including:
   - Complete descriptions (e.g., "deletion of the coding sequence (nt 1314 through nt 1328)")
   - Long descriptive phrases (e.g., "15 nucleotide (nt) deletion")
   - Complex entity names (e.g., "mutant protein variant X")
   - Entities in parentheses (e.g., "vascular endothelial growth factor (VEGF)")
   - Entities in brackets or with qualifiers
4. If an entity is described in multiple ways, use the MOST COMPLETE description available
5. Pay special attention to mutations, variants, and complex biomedical descriptions
6. Include ALL words that are part of the entity description (don't shorten)
7. If an entity has both a full name and abbreviation, use the FULL name if both appear, otherwise use what appears in the text
8. When extracting, verify the EXACT text appears in the source document

"""
        
        prompt += f"Text:\n{text}\n\n"
        
        return prompt
    
    def _get_relation_type_definitions(self) -> str:
        """Get relation type definitions for prompts."""
        if not self.include_relation_types:
            return ""
        
        return """
RELATION TYPE DEFINITIONS:
You MUST use ONLY these 8 relation types. Do NOT create new relation types.
These are the ONLY relation types used in the gold standard dataset.

ALLOWED RELATION TYPES (use exactly as written - these are the ONLY types that exist):

1. Association
   - General association between entities
   - Default choice when specific type is unclear
   - Use for: general relationships, dependencies, connections, co-occurrences
   - Examples: "X is associated with Y", "X relates to Y", "X depends on Y", "X is required for Y"
   - Also use for: regulatory relationships (regulates, controls, modulates), causal relationships when unsure

2. Positive_Correlation
   - Positive correlation or positive relationship
   - Use when entities have a positive relationship or correlation
   - Examples: "X increases Y", "X enhances Y", "X promotes Y", "X leads to Y", "X causes Y", "X upregulates Y"
   - Also use for: mutations leading to diseases, genes causing effects, treatments improving conditions

3. Negative_Correlation
   - Negative correlation or inverse relationship
   - Use when entities have a negative relationship or inverse correlation
   - Examples: "X decreases Y", "X inhibits Y", "X suppresses Y", "X reduces Y", "X downregulates Y", "X prevents Y"
   - Also use for: treatments reducing symptoms, inhibitors blocking effects, mutations preventing function

4. Bind
   - Binding relationship between entities
   - Use when entities physically or functionally bind to each other
   - Examples: "protein X binds to protein Y", "drug binds to receptor", "X interacts with Y" (physical binding)
   - Use ONLY for physical/functional binding, not general interactions

5. Cotreatment
   - Co-treatment relationship
   - Use when entities are used together in treatment or therapy
   - Examples: "drug X and drug Y used together", "combination therapy with X and Y", "X combined with Y"
   - Use for treatment combinations, not general co-occurrence

6. Comparison
   - Comparative relationship
   - Use when comparing entities or their effects
   - Examples: "X compared to Y", "X versus Y", "comparison of X and Y", "X vs Y"
   - Use ONLY for explicit comparisons, not general associations

7. Drug_Interaction
   - Drug interaction relationship
   - Use for interactions between drugs or drug-drug interactions
   - Examples: "drug X interacts with drug Y", "drug-drug interaction between X and Y", "X has interaction with Y"
   - Use specifically for drug-drug interactions

8. Conversion
   - Conversion relationship
   - Use when one entity converts to another
   - Examples: "X converts to Y", "X is converted into Y", "metabolic conversion of X to Y", "X transforms into Y"
   - Use for transformations and conversions, including metabolic pathways

CRITICAL RULES:
1. These are the ONLY 8 relation types that exist in the dataset
2. DO NOT use any other relation types such as: "Causes", "Treats", "Regulates", "Part_Of", "Increases", "Decreases", "Prevents", "Characterized_By", "Associated_With", etc.
3. Map common phrases to allowed types:
   - "causes", "leads to", "results in" → Positive_Correlation
   - "inhibits", "suppresses", "prevents", "blocks" → Negative_Correlation
   - "regulates", "controls", "modulates" → Association or Positive_Correlation (if increase) or Negative_Correlation (if decrease)
   - "treats", "therapies" → Association or Positive_Correlation
   - "is part of", "contains" → Association
4. Choose the most specific type that fits, but only from the 8 allowed types above
5. When unsure about type, use "Association" - it's the safest choice
6. DO NOT invent new relation types
"""
    
    def _get_brief_relation_type_definitions(self) -> str:
        """Get brief relation type definitions for improved prompts (less verbose than full)."""
        return """
RELATION TYPES (use ONLY these 8 types):
1. Association - General association, default when unsure
2. Positive_Correlation - "increases", "causes", "leads to", "promotes"
3. Negative_Correlation - "decreases", "inhibits", "suppresses", "reduces"
4. Bind - Physical/functional binding between entities
5. Cotreatment - Entities used together in treatment
6. Comparison - Explicit comparison between entities
7. Drug_Interaction - Drug-drug interactions
8. Conversion - One entity converts to another

Map other phrases: "causes/leads to" → Positive_Correlation, "inhibits/prevents" → Negative_Correlation, "regulates" → Association
"""

    def _get_brief_few_shot_examples(self) -> str:
        """Get brief few-shot examples for improved prompts (fewer examples than full)."""
        return """
EXAMPLES:

Example 1:
Text: "Curcumin decreases Sp1 expression in bladder cancer cells."
Output:
[
  {"head_mention": "Curcumin", "tail_mention": "Sp1 expression", "relation_type": "Negative_Correlation"},
  {"head_mention": "Curcumin", "tail_mention": "bladder cancer", "relation_type": "Negative_Correlation"}
]

Example 2:
Text: "The deletion of nt 1314-1328 resulted in loss of function."
Output:
[
  {"head_mention": "deletion of nt 1314-1328", "tail_mention": "loss of function", "relation_type": "Positive_Correlation"}
]
Note: Use EXACT, COMPLETE entity descriptions from text.

Example 3:
Text: "Protein X binds to receptor Y and inhibits Gene Z."
Output:
[
  {"head_mention": "Protein X", "tail_mention": "receptor Y", "relation_type": "Bind"},
  {"head_mention": "Protein X", "tail_mention": "Gene Z", "relation_type": "Negative_Correlation"}
]
"""

    def _get_few_shot_examples(self) -> str:
        """Get few-shot examples for prompts."""
        return """
EXAMPLES OF CORRECT EXTRACTION (using ONLY the 8 allowed relation types):

Example 1 - Complex Entity Extraction:
Text: "The deletion of the coding sequence (nt 1314 through nt 1328) resulted in loss of function."
Extracted Relations:
[
  {
    "head_mention": "deletion of the coding sequence (nt 1314 through nt 1328)",
    "tail_mention": "loss of function",
    "relation_type": "Positive_Correlation"
  }
]
Note: Used the complete, exact entity description. Mapped "resulted in" to Positive_Correlation.

Example 2 - Negative Correlation:
Text: "Recent studies show that protein X negatively correlates with disease Y progression."
Extracted Relations:
[
  {
    "head_mention": "protein X",
    "tail_mention": "disease Y progression",
    "relation_type": "Negative_Correlation"
  }
]
Note: Used "Negative_Correlation" for negative relationship. Extracted even if novel.

Example 3 - Multiple Relations (using allowed types only):
Text: "Gene A regulates Gene B, and Gene B causes Disease C."
Extracted Relations:
[
  {
    "head_mention": "Gene A",
    "tail_mention": "Gene B",
    "relation_type": "Positive_Correlation"
  },
  {
    "head_mention": "Gene B",
    "tail_mention": "Disease C",
    "relation_type": "Positive_Correlation"
  }
]
Note: Mapped "regulates" and "causes" to Positive_Correlation (allowed types only). Extract ALL relations.

Example 4 - Association (default):
Text: "The 15 nucleotide (nt) deletion in the coding region affects protein function."
Extracted Relations:
[
  {
    "head_mention": "15 nucleotide (nt) deletion in the coding region",
    "tail_mention": "protein function",
    "relation_type": "Association"
  }
]
Note: Used "Association" for general relationship. Used complete entity description.

Example 5 - Binding Relationship:
Text: "Protein X binds to receptor Y with high affinity."
Extracted Relations:
[
  {
    "head_mention": "Protein X",
    "tail_mention": "receptor Y",
    "relation_type": "Bind"
  }
]
Note: Used "Bind" for binding relationships.

Example 6 - Comprehensive Extraction:
Text: "Curcumin decreases Sp1 expression and inhibits bladder cancer cell growth. Sp1 regulates VEGF expression."
Extracted Relations:
[
  {
    "head_mention": "Curcumin",
    "tail_mention": "Sp1 expression",
    "relation_type": "Negative_Correlation"
  },
  {
    "head_mention": "Curcumin",
    "tail_mention": "bladder cancer cell growth",
    "relation_type": "Negative_Correlation"
  },
  {
    "head_mention": "Sp1",
    "tail_mention": "VEGF expression",
    "relation_type": "Positive_Correlation"
  }
]
Note: Extracted ALL three relations. Mapped "decreases" to Negative_Correlation, "inhibits" to Negative_Correlation, "regulates" to Positive_Correlation.

CRITICAL REMINDERS:
- Extract ALL relations, even if they seem obvious or implicit
- Use the MOST COMPLETE entity descriptions available (include full phrases, not abbreviations)
- Only extract relations that are EXPLICITLY stated in the text (do not infer)
- Verify each entity mention appears exactly in the source text (use exact text spans)
- Use ONLY the 8 allowed relation types - map any other concepts to these types
- When multiple relations exist between entities, extract ALL of them
- Pay special attention to relations involving complex entity descriptions (mutations, variants, etc.)
- Review the entire text multiple times to ensure nothing is missed
"""
    
    def _get_entity_context(self) -> Optional[str]:
        """
        Get entity context from global entity map for prompting.
        
        Returns:
            Entity context string or None if entity_map is not available
        """
        if not self.entity_map:
            return None
        
        # Get some common entities to provide context
        # This can be customized by subclasses
        return None
    
    def _make_api_request_with_retry(
        self,
        url: str,
        headers: dict,
        payload: dict,
        timeout: int = 180,
        max_retries: int = 10,
        base_delay: float = 2.0,
    ) -> requests.Response:
        """
        Make API request with retry logic and exponential backoff.
        
        Rate limit (429) errors will retry indefinitely until success.
        Other errors will retry up to max_retries times.
        
        Args:
            url: API endpoint URL
            headers: Request headers
            payload: Request payload
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts (not applied to 429 errors)
            base_delay: Base delay in seconds for exponential backoff
            
        Returns:
            Response object
            
        Raises:
            RuntimeError: If all retry attempts fail (except for rate limits which retry forever)
        """
        last_exception = None
        rate_limit_attempt = 0  # Track rate limit retries separately
        attempt = 0
        
        while True:
            try:
                if attempt > 0 and rate_limit_attempt == 0:
                    # Exponential backoff for non-rate-limit retries
                    delay = base_delay * (2 ** (attempt - 1))
                    self.logger.warning(
                        f"[{self.name}] Retry attempt {attempt}/{max_retries} "
                        f"after {delay:.1f}s delay..."
                    )
                    time.sleep(delay)
                
                self.logger.debug(
                    f"[{self.name}] API request attempt {attempt + 1} "
                    f"(timeout={timeout}s)"
                )
                
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
                response.raise_for_status()
                return response
                
            except requests.exceptions.Timeout as e:
                last_exception = e
                self.logger.warning(
                    f"[{self.name}] Request timeout (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )
                attempt += 1
                if attempt <= max_retries:
                    continue
                else:
                    break
                    
            except requests.exceptions.RequestException as e:
                last_exception = e
                # For non-timeout errors, only retry if it's a 5xx server error or 429 rate limit
                if hasattr(e, 'response') and e.response is not None:
                    status_code = e.response.status_code
                    if 500 <= status_code < 600:
                        self.logger.warning(
                            f"[{self.name}] Server error {status_code} "
                            f"(attempt {attempt + 1}/{max_retries + 1}): {e}"
                        )
                        attempt += 1
                        if attempt <= max_retries:
                            continue
                        else:
                            break
                    elif status_code == 429:
                        # Rate limit error - retry indefinitely until success
                        rate_limit_attempt += 1
                        error_msg = str(e)
                        retry_after = None
                        
                        # Try to extract retry-after time from error message
                        try:
                            error_details = e.response.json()
                            error_msg = error_details.get('error', {}).get('message', str(e))
                            # Extract retry time from message like "Please try again in 4.336s"
                            match = re.search(r'Please try again in ([\d.]+)s', error_msg)
                            if match:
                                retry_after = float(match.group(1)) + 1.0  # Add 1s buffer
                        except:
                            pass
                        
                        # Use retry_after if available, otherwise use exponential backoff
                        if retry_after:
                            delay = retry_after
                        else:
                            # Exponential backoff: 5s, 10s, 20s, 40s, 80s, capped at 120s
                            delay = min(5.0 * (2 ** min(rate_limit_attempt - 1, 5)), 120.0)
                        
                        self.logger.warning(
                            f"[{self.name}] Rate limit (429) exceeded. "
                            f"Retrying after {delay:.1f}s (rate limit retry #{rate_limit_attempt}, will keep retrying...)"
                        )
                        if error_msg and rate_limit_attempt == 1:  # Only log error message once
                            self.logger.warning(f"[{self.name}] Rate limit details: {error_msg[:200]}")
                        time.sleep(delay)
                        # Don't increment attempt counter - rate limits retry forever
                        continue
                    elif status_code == 400:
                        # 400 errors usually mean invalid model or request - don't retry
                        # Try to get error details from response
                        error_msg = str(e)
                        try:
                            error_details = e.response.json()
                            error_msg = error_details.get('error', {}).get('message', str(e))
                            error_type = error_details.get('error', {}).get('type', 'unknown')
                            self.logger.error(
                                f"[{self.name}] Bad Request (400): {error_type} - {error_msg}"
                            )
                            self.logger.error(
                                f"[{self.name}] Full error response: {error_details}"
                            )
                        except Exception as parse_error:
                            self.logger.error(f"[{self.name}] Bad Request (400): {e}")
                            self.logger.error(f"[{self.name}] Could not parse error response: {parse_error}")
                            # Try to get raw response text
                            try:
                                raw_response = e.response.text
                                self.logger.error(f"[{self.name}] Raw response: {raw_response}")
                                error_msg = raw_response if raw_response else str(e)
                            except:
                                pass
                        raise RuntimeError(f"API error (400 Bad Request): {error_msg}")
                    else:
                        # Don't retry for other client errors (4xx)
                        # Try to extract error message from response
                        error_msg = str(e)
                        try:
                            error_details = e.response.json()
                            error_msg = error_details.get('error', {}).get('message', str(e))
                            error_type = error_details.get('error', {}).get('type', 'unknown')
                            self.logger.error(
                                f"[{self.name}] Client error {status_code}: {error_type} - {error_msg[:300]}"
                            )
                        except Exception:
                            self.logger.error(f"[{self.name}] Client error {status_code}: {str(e)[:300]}")
                        raise RuntimeError(f"API error ({status_code}): {error_msg[:300]}")
                else:
                    # Network errors - retry
                    self.logger.warning(
                        f"[{self.name}] Network error (attempt {attempt + 1}/{max_retries + 1}): {e}"
                    )
                    attempt += 1
                    if attempt <= max_retries:
                        continue
                    else:
                        break
        
        # All retries exhausted
        error_msg = f"API request failed after {max_retries + 1} attempts"
        if last_exception:
            error_msg += f": {last_exception}"
        self.logger.error(f"[{self.name}] {error_msg}")
        raise RuntimeError(error_msg)
    
    def _extract_response_from_responses_api(self, result: dict) -> str:
        """
        Extract LLM response content from Responses API result structure.
        
        This handles GPT-5.x models that use the /responses endpoint, which
        may return responses in different formats including reasoning type responses.
        
        Args:
            result: JSON response from Responses API
            
        Returns:
            Extracted response string, or empty string if no content found
        """
        import json
        
        # Helper function to extract string content from nested structures
        def extract_content(value):
            """Recursively extract string content from nested lists/dicts."""
            if isinstance(value, str):
                return value
            elif isinstance(value, list):
                # Extract content from each item and join
                parts = []
                for item in value:
                    extracted = extract_content(item)
                    if extracted and extracted.strip():
                        parts.append(extracted)
                return "\n".join(parts) if parts else None
            elif isinstance(value, dict):
                # For GPT-5 Responses API, check for reasoning type responses
                # These may have the content in different fields
                response_type = value.get("type", "")
                
                # Try common content fields first
                for field in ["output", "content", "text", "message", "items", "summary"]:
                    if field in value:
                        extracted = extract_content(value[field])
                        if extracted and extracted.strip():
                            return extracted
                
                # For reasoning type responses, check if there's content in nested structures
                if response_type == "reasoning":
                    # Check for items array which might contain the actual output
                    if "items" in value:
                        extracted = extract_content(value["items"])
                        if extracted and extracted.strip():
                            return extracted
                    # Check for any nested dicts that might contain content
                    for key, val in value.items():
                        if key not in ["id", "type"] and isinstance(val, (list, dict)):
                            extracted = extract_content(val)
                            if extracted and extracted.strip():
                                return extracted
                
                # Try to find any string values in the dict (excluding metadata fields)
                for key, val in value.items():
                    if key not in ["id", "type", "status", "created"]:
                        if isinstance(val, str) and val.strip():
                            return val
                        elif isinstance(val, (list, dict)):
                            extracted = extract_content(val)
                            if extracted and extracted.strip():
                                return extracted
                
                # Fallback: return None to indicate no content found
                return None
            else:
                return str(value) if value else None
        
        # Log the response structure for debugging
        self.logger.debug(f"[{self.name}] Responses API result keys: {list(result.keys())}")
        self.logger.debug(f"[{self.name}] Responses API result type: {result.get('type', 'N/A')}")
        
        # Check if result itself is a response object (has 'type' and 'id' fields)
        # This might indicate the response is returned directly, not wrapped
        llm_response = None
        if "type" in result and "id" in result:
            self.logger.debug(f"[{self.name}] Response object detected with type: {result.get('type')}")
            # Try to extract from the response object itself
            llm_response = extract_content(result)
        
        # First, try the standard "output" field
        if not llm_response or not llm_response.strip():
            if "output" in result:
                llm_response = extract_content(result["output"])
        
        # If output is a list (common in Responses API), extract from items
        if not llm_response or not llm_response.strip():
            if isinstance(result.get("output"), list) and len(result["output"]) > 0:
                # Extract from first item if it's a list
                llm_response = extract_content(result["output"][0])
        
        # Try choices array (fallback for compatibility)
        if not llm_response or not llm_response.strip():
            if "choices" in result and len(result["choices"]) > 0:
                output = result["choices"][0].get("output", result["choices"][0].get("message", {}).get("content", ""))
                llm_response = extract_content(output)
        
        # Try extracting from top-level fields
        if not llm_response or not llm_response.strip():
            for key in ["output", "response", "content", "text", "items", "summary"]:
                if key in result:
                    llm_response = extract_content(result[key])
                    if llm_response and llm_response.strip():
                        break
        
        # If still no response, log the full structure for debugging
        if not llm_response or not llm_response.strip():
            error_msg = f"Unexpected response structure from Responses API. Response type: {result.get('type', 'unknown')}, Keys: {list(result.keys())}"
            self.logger.error(f"[{self.name}] {error_msg}")
            self.logger.debug(f"[{self.name}] Full API response: {json.dumps(result, indent=2)[:2000]}")
            # For reasoning type responses with empty summary, this might be expected
            # Return empty string so pipeline can continue (will result in 0 relations)
            if result.get("type") == "reasoning" and result.get("summary") == []:
                self.logger.warning(f"[{self.name}] GPT-5 returned reasoning type response with empty summary. This may indicate the model did not generate output.")
            llm_response = ""
        
        return llm_response