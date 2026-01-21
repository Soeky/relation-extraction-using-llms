"""Text-based relation matcher that compares entity mentions directly.

This matcher converts gold relations (with entity IDs) to text mentions and compares
them directly with LLM output text. This avoids entity resolution issues and provides
more accurate evaluation.
"""

import logging
import re
import string
from typing import List, Tuple, Optional, Set
from difflib import SequenceMatcher

from ..types import Relation, ParsedRelation, GoldRelations

# Try to import BERTScore (optional)
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False


class TextRelationMatcher:
    """Matches relations by comparing entity mention texts directly."""
    
    def __init__(
        self,
        use_bertscore: bool = False,
        similarity_threshold: float = 0.85,
        fuzzy_threshold: float = 0.7,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize text-based relation matcher.
        
        Args:
            use_bertscore: Whether to use BERTScore for semantic similarity
            similarity_threshold: Minimum similarity for BERTScore matching (0-1)
            fuzzy_threshold: Minimum similarity for fuzzy string matching (0-1)
            logger: Optional logger instance
        """
        self.use_bertscore = use_bertscore and BERTSCORE_AVAILABLE
        self.similarity_threshold = similarity_threshold
        self.fuzzy_threshold = fuzzy_threshold
        self.logger = logger or logging.getLogger(__name__)
        
        if use_bertscore and not BERTSCORE_AVAILABLE:
            self.logger.warning(
                "[TextRelationMatcher] BERTScore requested but not available. "
                "Falling back to fuzzy string matching."
            )
    
    def convert_gold_relation_to_text(
        self,
        gold_rel: Relation,
        gold_relations_obj: GoldRelations
    ) -> List[Tuple[str, str, str]]:
        """
        Convert a gold relation to all possible text-based representations.
        
        A gold relation can have multiple text representations because:
        - Each entity can have multiple mentions
        - We need to try all combinations
        
        Args:
            gold_rel: Gold relation with entity IDs
            gold_relations_obj: GoldRelations object containing entities
            
        Returns:
            List of (head_mention, tail_mention, relation_type) tuples
        """
        # Find head and tail entities
        head_entity = None
        tail_entity = None
        
        for entity in gold_relations_obj.entities:
            if entity.id == gold_rel.head_id:
                head_entity = entity
            if entity.id == gold_rel.tail_id:
                tail_entity = entity
        
        if not head_entity or not tail_entity:
            # Can't convert if entities not found
            return []
        
        # Get all mentions for head and tail entities
        head_mentions = [m.text for m in head_entity.mentions] if head_entity.mentions else []
        tail_mentions = [m.text for m in tail_entity.mentions] if tail_entity.mentions else []
        
        # If no mentions, try to use entity ID as fallback (not ideal but better than nothing)
        if not head_mentions:
            head_mentions = [gold_rel.head_id]
        if not tail_mentions:
            tail_mentions = [gold_rel.tail_id]
        
        # Generate all combinations
        text_relations = []
        for head_mention in head_mentions:
            for tail_mention in tail_mentions:
                text_relations.append((head_mention, tail_mention, gold_rel.type))
        
        return text_relations
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for better matching.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Lowercase
        normalized = text.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove punctuation (optional - can be made configurable)
        # normalized = normalized.translate(str.maketrans('', '', string.punctuation))
        
        return normalized
    
    def _extract_core_phrase(self, text: str) -> str:
        """
        Extract core phrase by removing leading numbers/quantifiers.
        
        This helps match "15 nucleotide (nt) deletion..." with "deletion..."
        
        Args:
            text: Text to extract core from
            
        Returns:
            Core phrase
        """
        normalized = self._normalize_text(text)
        
        # Remove leading numbers and common quantifiers
        # Pattern: number + optional unit + optional parentheses + text
        # e.g., "15 nucleotide (nt) deletion..." -> "deletion..."
        pattern = r'^\d+\s*(?:nucleotide|nt|amino\s*acid|aa|base\s*pair|bp)?\s*(?:\([^)]+\))?\s*'
        core = re.sub(pattern, '', normalized, flags=re.IGNORECASE)
        
        # If we removed something, return the core; otherwise return original
        if core != normalized and len(core) > 10:  # Only if we got a meaningful core
            return core.strip()
        
        return normalized
    
    def compute_text_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Compute similarity between two texts.
        
        Uses BERTScore if available, otherwise fuzzy string matching with normalization.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts
        norm1 = self._normalize_text(text1)
        norm2 = self._normalize_text(text2)
        
        # Exact match after normalization
        if norm1 == norm2:
            return 1.0
        
        if self.use_bertscore:
            try:
                P, R, F1 = bert_score(
                    [text1],
                    [text2],
                    lang="en",
                    verbose=False
                )
                return float(F1.item())
            except Exception as e:
                self.logger.warning(f"[TextRelationMatcher] BERTScore error: {e}, using fuzzy matching")
        
        # Fuzzy string matching on normalized text
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        
        # Also check if one contains the other (for partial matches)
        # This helps with cases like "Iodide transport defect (ITD)" vs "Iodide transport defect"
        if norm1 in norm2 or norm2 in norm1:
            # Boost similarity for containment
            similarity = max(similarity, 0.85)
        
        # Try extracting core phrases (removes leading numbers/quantifiers)
        # This helps match "15 nucleotide (nt) deletion..." with "deletion..."
        core1 = self._extract_core_phrase(text1)
        core2 = self._extract_core_phrase(text2)
        
        if core1 != norm1 or core2 != norm2:
            # We extracted cores, check if they match better
            core_similarity = SequenceMatcher(None, core1, core2).ratio()
            if core1 in core2 or core2 in core1:
                core_similarity = max(core_similarity, 0.90)  # Higher boost for core match
            similarity = max(similarity, core_similarity)
        
        return similarity
    
    def match_relation_texts(
        self,
        pred_head: str,
        pred_tail: str,
        pred_type: str,
        gold_text_relations: List[Tuple[str, str, str]],
        match_type: bool = True
    ) -> Tuple[Optional[Tuple[str, str, str]], float]:
        """
        Match a predicted relation to gold text relations.
        
        Args:
            pred_head: Predicted head entity mention
            pred_tail: Predicted tail entity mention
            pred_type: Predicted relation type
            gold_text_relations: List of (head, tail, type) tuples from gold
            match_type: Whether to consider relation type in matching (default: True)
            
        Returns:
            Tuple of (best_match, similarity_score) or (None, 0.0) if no match
        """
        best_match = None
        best_score = 0.0
        
        for gold_head, gold_tail, gold_type in gold_text_relations:
            # Compute head similarity
            head_sim = self.compute_text_similarity(pred_head, gold_head)
            
            # Compute tail similarity
            tail_sim = self.compute_text_similarity(pred_tail, gold_tail)
            
            # Check if relation types match (case-insensitive)
            type_match = 1.0 if (not match_type) or (pred_type.upper() == gold_type.upper()) else 0.0
            
            # Combined score: average of head, tail, and type (if match_type is True)
            if match_type:
                # Weight: 40% head, 40% tail, 20% type
                combined_score = (0.4 * head_sim) + (0.4 * tail_sim) + (0.2 * type_match)
            else:
                # Entity-only matching: 50% head, 50% tail
                combined_score = (0.5 * head_sim) + (0.5 * tail_sim)
            
            # Also check reverse direction (tail-head)
            reverse_head_sim = self.compute_text_similarity(pred_head, gold_tail)
            reverse_tail_sim = self.compute_text_similarity(pred_tail, gold_head)
            if match_type:
                reverse_score = (0.4 * reverse_head_sim) + (0.4 * reverse_tail_sim) + (0.2 * type_match)
            else:
                reverse_score = (0.5 * reverse_head_sim) + (0.5 * reverse_tail_sim)
            
            # Take the better of forward or reverse
            score = max(combined_score, reverse_score)
            
            if score > best_score:
                best_score = score
                best_match = (gold_head, gold_tail, gold_type)
        
        # Check if score meets threshold
        threshold = self.similarity_threshold if self.use_bertscore else self.fuzzy_threshold
        if best_score >= threshold:
            return best_match, best_score
        
        return None, best_score
    
    def match(
        self,
        predicted_relations: List[ParsedRelation],
        gold_relations_obj: GoldRelations,
        match_type: bool = True
    ) -> Tuple[List[Relation], List[ParsedRelation], List[Relation], List[Tuple[ParsedRelation, Relation, float]]]:
        """
        Match predicted relations to gold relations using text-based comparison.
        
        This method converts gold relations (with entity IDs) to text mentions and
        compares them directly with LLM output text. This avoids entity resolution
        issues and provides more accurate evaluation.
        
        Args:
            predicted_relations: List of predicted relations with text mentions
            gold_relations_obj: GoldRelations object containing gold relations and entities
            match_type: Whether to require relation type to match (default: True)
            
        Returns:
            Tuple of (true_positives, false_positives, false_negatives, semantic_matches)
            where semantic_matches are (predicted, gold, similarity_score) tuples
        """
        # Convert all gold relations to text-based representations
        # Use relation IDs as keys since Relation objects are not hashable
        gold_text_relations_map = {}  # Dict[str, List[Tuple[str, str, str]]]
        gold_relations_by_id = {}  # Dict[str, Relation] - mapping from ID to Relation object
        for gold_rel in gold_relations_obj.relations:
            text_relations = self.convert_gold_relation_to_text(gold_rel, gold_relations_obj)
            if text_relations:
                gold_text_relations_map[gold_rel.id] = text_relations
                gold_relations_by_id[gold_rel.id] = gold_rel
        
        self.logger.debug(
            f"[TextRelationMatcher] Converted {len(gold_text_relations_map)} gold relations "
            f"to text representations (match_type={match_type})"
        )
        
        matched_gold_relation_ids = set()  # Set of relation IDs (hashable)
        true_positives = []
        false_positives = []
        semantic_matches = []  # Relations matched by text similarity
        
        # Match each predicted relation
        for pred_rel in predicted_relations:
            best_gold_rel_id = None
            best_score = 0.0
            best_text_match = None
            
            # Try to match against all gold relations
            for gold_rel_id, gold_text_relations in gold_text_relations_map.items():
                if gold_rel_id in matched_gold_relation_ids:
                    continue
                
                # Try to match text
                text_match, score = self.match_relation_texts(
                    pred_rel.head_mention,
                    pred_rel.tail_mention,
                    pred_rel.relation_type,
                    gold_text_relations,
                    match_type=match_type
                )
                
                if score > best_score:
                    best_score = score
                    best_gold_rel_id = gold_rel_id
                    best_text_match = text_match
            
            # Check if we found a good match
            threshold = self.similarity_threshold if self.use_bertscore else self.fuzzy_threshold
            
            if best_gold_rel_id and best_score >= threshold:
                best_gold_rel = gold_relations_by_id[best_gold_rel_id]
                
                # Check if relation types match (if match_type is True, case-insensitive)
                type_matches = (not match_type) or (pred_rel.relation_type.upper() == best_gold_rel.type.upper())
                
                # Check if it's an exact match (similarity above threshold + type matches if required)
                is_exact = (
                    best_score >= 0.70 and  # Similarity above threshold (0.7 for fuzzy, 0.85 for BERTScore)
                    type_matches
                )
                
                if is_exact:
                    true_positives.append(best_gold_rel)
                    matched_gold_relation_ids.add(best_gold_rel_id)
                elif type_matches:
                    # Semantic match (similar entities, type matches)
                    semantic_matches.append((pred_rel, best_gold_rel, best_score))
                    matched_gold_relation_ids.add(best_gold_rel_id)
                else:
                    # Entities match but type doesn't (only if match_type is True)
                    # Still count as semantic match for entity-only evaluation
                    semantic_matches.append((pred_rel, best_gold_rel, best_score))
                    matched_gold_relation_ids.add(best_gold_rel_id)
            else:
                # No match found
                false_positives.append(pred_rel)
        
        # False negatives: unmatched gold relations
        false_negatives = [
            gold_rel for gold_rel in gold_relations_obj.relations
            if gold_rel.id not in matched_gold_relation_ids
        ]
        
        return true_positives, false_positives, false_negatives, semantic_matches

