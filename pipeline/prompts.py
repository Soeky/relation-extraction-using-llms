"""
Prompt templates for relation extraction.

This file contains all prompt templates used by different prompting techniques (IO, CoT, RAG, ReAct).
Edit the prompts here to adjust them for better performance.
"""

# ============================================================================
# TRAIN DATA EXAMPLE (for reference)
# ============================================================================
"""
Example from train dataset (doc_id: 18593936) - use this as reference for prompt improvements.

TEXT:
---
Curcumin decreases specificity protein expression in bladder cancer cells.

Curcumin is the active component of tumeric, and this polyphenolic compound has been extensively 
investigated as an anticancer drug that modulates multiple pathways and genes. In this study, 10 to 
25 micromol/L curcumin inhibited 253JB-V and KU7 bladder cancer cell growth, and this was 
accompanied by induction of apoptosis and decreased expression of the proapoptotic protein survivin 
and the angiogenic proteins vascular endothelial growth factor (VEGF) and VEGF receptor 1 (VEGFR1). 
Because expression of survivin, VEGF, and VEGFR1 are dependent on specificity protein (Sp) 
transcription factors, we also investigated the effects of curcumin on Sp protein expression as an 
underlying mechanism for the apoptotic and antiangiogenic activity of this compound. The results 
show that curcumin induced proteasome-dependent down-regulation of Sp1, Sp3, and Sp4 in 253JB-V and 
KU7 cells. Moreover, using RNA interference with small inhibitory RNAs for Sp1, Sp3, and Sp4, we 
observed that curcumin-dependent inhibition of nuclear factor kappaB (NF-kappaB)-dependent genes, 
such as bcl-2, survivin, and cyclin D1, was also due, in part, to loss of Sp proteins. Curcumin 
also decreased bladder tumor growth in athymic nude mice bearing KU7 cells as xenografts and this 
was accompanied by decreased Sp1, Sp3, and Sp4 protein levels in tumors. These results show for 
the first time that one of the underlying mechanisms of action of curcumin as a cancer 
chemotherapeutic agent is due, in part, to decreased expression of Sp transcription factors in 
bladder cancer cells.
---

GOLD RELATIONS:
- Curcumin -> Negative_Correlation -> bladder cancer (novel)
- Curcumin -> Negative_Correlation -> Sp1, Sp3, Sp4 (novel)
- Curcumin -> Negative_Correlation -> survivin, VEGF, VEGFR1 (novel)
- Curcumin -> Negative_Correlation -> bcl-2, cyclin D1 (novel)
- Sp1, Sp3, Sp4 -> Association -> bladder cancer (both novel and non-novel)
- proteasome -> Association -> Sp1, Sp3, Sp4 (non-novel)
- NF-kappaB -> Association -> bcl-2, survivin, cyclin D1 (non-novel)
- survivin, VEGF, VEGFR1 -> Association -> Sp transcription factors (non-novel)
... and many more

Key observations:
- Many relations involve complex entity descriptions (e.g., "specificity protein (Sp) transcription factors")
- Mix of explicit relations (Curcumin decreases X) and implicit dependencies (X is dependent on Y)
- Both novel and non-novel relations present
- Some entities appear in multiple forms (e.g., "vascular endothelial growth factor" vs "VEGF")
- Relations can be between different entity types: ChemicalEntity, Disease, GeneOrGeneProduct, etc.
"""


# ============================================================================
# BASE PROMPT TEMPLATES
# ============================================================================

def get_base_prompt_intro() -> str:
    """Get the base introduction for prompts."""
    return """Extract biomedical relations from the following text.

"""


def get_entity_extraction_instructions() -> str:
    """Get instructions for entity extraction."""
    return """CRITICAL INSTRUCTIONS FOR ENTITY EXTRACTION:
1. Use the EXACT text spans from the document for all entity mentions
2. Do NOT paraphrase, summarize, or modify entity mentions
3. Copy entity mentions exactly as they appear in the text, including:
   - Complete descriptions (e.g., "deletion of the coding sequence (nt 1314 through nt 1328)")
   - Long descriptive phrases (e.g., "15 nucleotide (nt) deletion")
   - Complex entity names (e.g., "mutant protein variant X")
   - Abbreviations and their full forms only if they appear in the text (e.g., "VEGF" and "vascular endothelial growth factor")
4. If an entity is described in multiple ways, use the MOST COMPLETE description available
5. Pay special attention to mutations, variants, and complex biomedical descriptions
6. Include entities mentioned in parentheses, brackets, or complex phrases

"""


def get_relation_type_definitions() -> str:
    """Get relation type definitions for prompts."""
    return """
RELATION TYPE DEFINITIONS:
You MUST use ONLY these 8 relation types. Do NOT create new relation types.
These are the ONLY relation types used in the gold standard dataset.

ALLOWED RELATION TYPES (use exactly as written - these are the ONLY types that exist):

1. Association
   - General association between entities
   - Default choice when specific type is unclear
   - Use for: general relationships, dependencies, connections, co-occurrences
   - Examples: "X is associated with Y", "X relates to Y", "X depends on Y"

2. Positive_Correlation
   - Positive correlation or positive relationship
   - Use when entities have a positive relationship or correlation
   - Examples: "X increases Y", "X enhances Y", "X promotes Y", "X leads to Y"
   - Also use for: mutations leading to diseases, genes causing effects

3. Negative_Correlation
   - Negative correlation or inverse relationship
   - Use when entities have a negative relationship or inverse correlation
   - Examples: "X decreases Y", "X inhibits Y", "X suppresses Y", "X reduces Y"
   - Also use for: treatments reducing symptoms, inhibitors blocking effects

4. Bind
   - Binding relationship between entities
   - Use when entities physically or functionally bind to each other
   - Examples: "protein X binds to protein Y", "drug binds to receptor"

5. Cotreatment
   - Co-treatment relationship
   - Use when entities are used together in treatment
   - Examples: "drug X and drug Y used together", "combination therapy"

6. Comparison
   - Comparative relationship
   - Use when comparing entities or their effects
   - Examples: "X compared to Y", "X versus Y", comparative studies

7. Drug_Interaction
   - Drug interaction relationship
   - Use for interactions between drugs or drug-drug interactions
   - Examples: "drug X interacts with drug Y", drug interaction effects

8. Conversion
   - Conversion relationship
   - Use when one entity converts to another
   - Examples: "X converts to Y", metabolic conversions, transformations

CRITICAL RULES:
1. These are the ONLY 8 relation types that exist in the dataset
2. DO NOT use any other relation types such as: "Causes", "Treats", "Regulates", "Part_Of", "Increases", "Decreases", "Prevents", etc.
3. Choose the most specific type that fits, but only from the 8 allowed types above
4. DO NOT invent new relation types
"""


def get_few_shot_examples() -> str:
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
Note: Used the complete, exact entity description. Used "Positive_Correlation" for causal relationship.

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
Note: Map "regulates" and "causes" to "Positive_Correlation". Extract ALL relations.

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

CRITICAL REMINDERS:
- Extract ALL relations, even if they seem obvious or implicit
- Use the MOST COMPLETE entity descriptions available
- Only extract relations that are EXPLICITLY stated in the text (do not infer)
- Verify each entity mention appears exactly in the source text
- Use ONLY the 8 allowed relation types - map any other concepts to these types
"""


# ============================================================================
# IO (INPUT/OUTPUT) PROMPT
# ============================================================================

def get_io_task_instructions() -> str:
    """Get task instructions for IO prompting."""
    return """TASK: Extract ALL biomedical relations from the text above.

INSTRUCTIONS:
1. Extract ALL relations, including those involving complex entity descriptions
2. Pay special attention to novel or newly discovered relations
3. Use the most complete entity mention available in the text
4. Only extract relations that are EXPLICITLY stated in the text (do not infer)
5. Before finalizing, verify each entity mention appears exactly in the text

"""


def get_io_output_format() -> str:
    """Get output format instructions for IO prompting."""
    return """
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
- Use EXACT text spans from the document for entity mentions
- Do not paraphrase or modify the text
- Extract ALL relations, even if they seem obvious
- Pay special attention to relations involving mutations, variants, and complex descriptions
- When unsure, use "Association" - it's the safest choice
"""


# ============================================================================
# CHAIN OF THOUGHT (COT) PROMPT
# ============================================================================

def get_cot_reasoning_steps() -> str:
    """Get reasoning steps for Chain of Thought prompting."""
    return """TASK: Extract ALL biomedical relations from the text above using a step-by-step reasoning approach.

REASONING STEPS:

Step 1: IDENTIFY ALL ENTITIES
- Scan the entire text for ALL biomedical entities (genes, proteins, diseases, drugs, mutations, etc.)
- For each entity, extract the EXACT text span from the document
- Pay special attention to complex entity mentions (mutations, variants, long descriptions)
- Include entities mentioned in parentheses, brackets, or complex phrases
- Think: "What are ALL the entities mentioned, including complex descriptions?"

Step 2: CLASSIFY ENTITY TYPES
- Classify each identified entity's type (e.g., DiseaseOrPhenotypicFeature, GeneOrGeneProduct, OrganismTaxon, ChemicalEntity)
- This helps understand the context but is not required for the final output

Step 3: IDENTIFY ALL RELATIONS
- For each pair of entities, determine if there is a relation
- Extract ALL relations, including:
  * Obvious relations (explicitly stated)
  * Implicit relations (strongly implied)
  * Novel relations (newly discovered or unexpected)
- For each relation, determine:
  * Head entity (use the exact text span from Step 1)
  * Tail entity (use the exact text span from Step 1)
  * Relation type (choose the most specific type that accurately describes the relationship)

Step 4: VERIFY RELATIONS
- Verify each relation is explicitly stated or strongly implied in the text
- Verify each entity mention appears exactly in the source text
- Check that relation types match the semantic meaning in the text
- Remove any relations that are inferred but not stated

Step 5: CHECK FOR MISSING RELATIONS
- Review the text again to ensure you haven't missed any relations
- Pay special attention to relations involving complex entity descriptions
- Check for novel or unexpected relations that might be easy to miss

"""


def get_cot_output_format() -> str:
    """Get output format for Chain of Thought prompting."""
    return """
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
- Use EXACT text spans from the document for entity mentions
- Do not paraphrase or modify the text
- Extract ALL relations, even if they seem obvious
- When unsure about type, use "Association"
"""


# ============================================================================
# RAG (RETRIEVAL AUGMENTED GENERATION) PROMPT
# ============================================================================

def get_rag_task_instructions() -> str:
    """Get task instructions for RAG prompting."""
    return """TASK: Extract ALL biomedical relations from the text above using the provided context.

INSTRUCTIONS:
1. Use the relevant context from PubMed to help identify entities and relations
2. Extract ALL relations, including those involving complex entity descriptions
3. Pay special attention to novel or newly discovered relations
4. Use the most complete entity mention available in the text
5. Only extract relations that are EXPLICITLY stated in the text (do not infer)
6. The context is provided to help understand domain knowledge, but relations must be from the main text

"""


def get_rag_output_format() -> str:
    """Get output format for RAG prompting."""
    return """
OUTPUT FORMAT:
Return the results as a JSON array with the following format:
[
  {
    "head_mention": "exact text from document",
    "tail_mention": "exact text from document",
    "relation_type": "Association"
  }
]

REMEMBER:
- Use EXACT text spans from the document for entity mentions
- Do not paraphrase or modify the text
- Extract ALL relations, even if they seem obvious
- The context helps with understanding, but all relations must come from the main text above
"""


# ============================================================================
# REACT (REASONING + ACTING) PROMPT
# ============================================================================

def get_react_reasoning_steps() -> str:
    """Get reasoning steps for ReAct prompting."""
    return """TASK: Extract ALL biomedical relations from the text above using a structured reasoning approach.

Follow these steps internally:

STEP 1: IDENTIFY ALL ENTITIES
- Find all entities (genes, proteins, diseases, drugs, mutations, etc.)
- Include entities with complex descriptions (e.g., "deletion of the coding sequence (nt 1314 through nt 1328)")
- Extract the EXACT text span for each entity
- Pay special attention to entities in parentheses, brackets, or complex phrases

STEP 2: IDENTIFY ALL RELATIONS
- Check all pairs of entities for potential relations
- Include relations that are explicitly stated
- Include relations that are strongly implied
- Pay special attention to novel or unexpected relations
- For each relation, identify: head entity (exact text), tail entity (exact text), and relation type

STEP 3: VERIFY RELATIONS
- Check each relation is explicitly stated (do not infer based on general knowledge)
- Verify each entity mention appears exactly in the source text
- Check that relation types match the semantic meaning in the text
- Remove any relations that are inferred but not stated

STEP 4: CHECK FOR MISSING RELATIONS
- Review the text again for any relations you might have missed
- Pay special attention to relations involving complex entity descriptions
- Look for novel or unexpected relations

"""


def get_react_output_format() -> str:
    """Get output format for ReAct prompting."""
    return """
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
- DO NOT use types like "Causes", "Treats", "Regulates", "Part_Of", "Characterized_By", "Associated_With", etc. - map them to allowed types
- Use EXACT text spans from the document for entity mentions
- Do not paraphrase or modify the text
- Extract ALL relations, even if they seem obvious
- When unsure about type, use "Association"
"""


# ============================================================================
# HELPER FUNCTIONS FOR BUILDING PROMPTS
# ============================================================================

def build_base_prompt(text: str, doc_id: str = None, use_exact_spans: bool = True) -> str:
    """
    Build base prompt with common instructions.
    
    Args:
        text: Document text
        doc_id: Optional document ID
        use_exact_spans: Whether to include entity extraction instructions
        
    Returns:
        Base prompt string
    """
    prompt = get_base_prompt_intro()
    
    if doc_id:
        prompt += f"Document ID: {doc_id}\n\n"
    
    if use_exact_spans:
        prompt += get_entity_extraction_instructions()
    
    prompt += f"Text:\n{text}\n\n"
    
    return prompt

