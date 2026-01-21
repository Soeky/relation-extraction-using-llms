"""Token-based matcher using rapidfuzz/thefuzz for relation matching."""

import logging
from typing import List, Tuple, Optional

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    try:
        from thefuzz import fuzz
        RAPIDFUZZ_AVAILABLE = True
    except ImportError:
        RAPIDFUZZ_AVAILABLE = False
        fuzz = None

from .base import BaseMatcher
from ...types import Relation, ParsedRelation, GoldRelations


class TokenMatcher(BaseMatcher):
    """Matches relations using token-based similarity (token sort/set ratio from rapidfuzz/thefuzz)."""
    
    def __init__(
        self,
        match_type: bool = True,
        similarity_threshold: float = 0.7,
        use_token_set: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize token matcher.
        
        Args:
            match_type: Whether to require relation type to match (default: True)
            similarity_threshold: Minimum similarity for a match (0-1)
            use_token_set: If True, use token_set_ratio; if False, use token_sort_ratio
            logger: Optional logger instance
        """
        super().__init__(match_type, similarity_threshold, logger)
        self.use_token_set = use_token_set
        
        if not RAPIDFUZZ_AVAILABLE:
            self.logger.warning(
                "[TokenMatcher] rapidfuzz/thefuzz not available. "
                "Install with: pip install rapidfuzz (or thefuzz). "
                "Falling back to basic string matching."
            )
    
    def get_name(self) -> str:
        """Get matcher name."""
        return "token"
    
    def get_config(self) -> dict:
        """Get matcher configuration."""
        return {
            "match_type": self.match_type,
            "similarity_threshold": self.similarity_threshold,
            "use_token_set": self.use_token_set,
            "method": "token_set_ratio" if self.use_token_set else "token_sort_ratio"
        }
    
    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute token-based similarity between two texts with containment checks.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1), boosted if containment detected
        """
        if not text1 or not text2:
            return 0.0
        
        # Normalize for containment check
        norm1 = self._normalize_text(text1)
        norm2 = self._normalize_text(text2)
        
        # Exact match after normalization
        if norm1 == norm2:
            return 1.0
        
        if not RAPIDFUZZ_AVAILABLE:
            # Fallback to basic normalization and equality
            return 1.0 if norm1 == norm2 else 0.0
        
        # Use rapidfuzz/thefuzz - get multiple similarity metrics
        if self.use_token_set:
            similarity = fuzz.token_set_ratio(text1, text2) / 100.0
        else:
            similarity = fuzz.token_sort_ratio(text1, text2) / 100.0
        
        # Also try partial ratio for better matching
        partial_sim = fuzz.partial_ratio(text1, text2) / 100.0
        
        # Use WRatio (weighted ratio) which combines multiple methods
        try:
            wrato_sim = fuzz.WRatio(text1, text2) / 100.0
        except:
            wrato_sim = similarity
        
        # Take max of all methods
        similarity = max(similarity, partial_sim, wrato_sim)
        
        # Check for containment (substring match) - boost similarity
        if norm1 in norm2 or norm2 in norm1:
            # Boost similarity for substring containment
            similarity = max(similarity, 0.80)
        
        # Check token containment (one text's tokens are subset of another's)
        try:
            tokens1 = set(self._normalize_text(text1).split())
            tokens2 = set(self._normalize_text(text2).split())
            if tokens1 and tokens2:
                if tokens1.issubset(tokens2) or tokens2.issubset(tokens1):
                    similarity = max(similarity, 0.75)
        except:
            pass
        
        return similarity
    
    def _convert_gold_relation_to_text(
        self,
        gold_rel: Relation,
        gold_relations_obj: GoldRelations
    ) -> List[Tuple[str, str, str]]:
        """Convert gold relation to text-based representations."""
        head_entity = None
        tail_entity = None
        
        for entity in gold_relations_obj.entities:
            if entity.id == gold_rel.head_id:
                head_entity = entity
            if entity.id == gold_rel.tail_id:
                tail_entity = entity
        
        if not head_entity or not tail_entity:
            return []
        
        head_mentions = [m.text for m in head_entity.mentions] if head_entity.mentions else []
        tail_mentions = [m.text for m in tail_entity.mentions] if tail_entity.mentions else []
        
        if not head_mentions:
            head_mentions = [gold_rel.head_id]
        if not tail_mentions:
            tail_mentions = [gold_rel.tail_id]
        
        text_relations = []
        for head_mention in head_mentions:
            for tail_mention in tail_mentions:
                text_relations.append((head_mention, tail_mention, gold_rel.type))
        
        return text_relations
    
    def match(
        self,
        predicted_relations: List[ParsedRelation],
        gold_relations_obj: GoldRelations,
        match_type: Optional[bool] = None
    ) -> Tuple[List[Relation], List[ParsedRelation], List[Relation], List[Tuple[ParsedRelation, Relation, float]]]:
        """Match predicted relations to gold relations using token-based similarity."""
        if match_type is None:
            match_type = self.match_type
        
        # Convert gold relations to text representations
        gold_text_relations_map = {}
        gold_relations_by_id = {}
        for gold_rel in gold_relations_obj.relations:
            text_relations = self._convert_gold_relation_to_text(gold_rel, gold_relations_obj)
            if text_relations:
                gold_text_relations_map[gold_rel.id] = text_relations
                gold_relations_by_id[gold_rel.id] = gold_rel
        
        matched_gold_relation_ids = set()
        true_positives = []
        false_positives = []
        semantic_matches = []
        
        # Match each predicted relation
        for pred_rel in predicted_relations:
            best_gold_rel_id = None
            best_score = 0.0
            
            for gold_rel_id, gold_text_relations in gold_text_relations_map.items():
                if gold_rel_id in matched_gold_relation_ids:
                    continue
                
                best_relation_score = 0.0
                
                for gold_head, gold_tail, gold_type in gold_text_relations:
                    head_sim = self.compute_text_similarity(pred_rel.head_mention, gold_head)
                    tail_sim = self.compute_text_similarity(pred_rel.tail_mention, gold_tail)
                    reverse_head_sim = self.compute_text_similarity(pred_rel.head_mention, gold_tail)
                    reverse_tail_sim = self.compute_text_similarity(pred_rel.tail_mention, gold_head)
                    
                    # Type match (case-insensitive)
                    type_match = 1.0 if (not match_type) or (pred_rel.relation_type.upper() == gold_type.upper()) else 0.0
                    
                    # Improved scoring: use better weighting that favors good matches
                    if match_type:
                        forward_score = 0.35 * head_sim + 0.35 * tail_sim + 0.3 * type_match
                        reverse_score = 0.35 * reverse_head_sim + 0.35 * reverse_tail_sim + 0.3 * type_match
                    else:
                        # Entity-only: use average but boost if one entity is well-matched
                        avg_forward = (head_sim + tail_sim) / 2.0
                        max_forward = max(head_sim, tail_sim)
                        forward_score = 0.6 * avg_forward + 0.4 * max_forward
                        
                        avg_reverse = (reverse_head_sim + reverse_tail_sim) / 2.0
                        max_reverse = max(reverse_head_sim, reverse_tail_sim)
                        reverse_score = 0.6 * avg_reverse + 0.4 * max_reverse
                    
                    relation_score = max(forward_score, reverse_score)
                    best_relation_score = max(best_relation_score, relation_score)
                
                if best_relation_score > best_score:
                    best_score = best_relation_score
                    best_gold_rel_id = gold_rel_id
            
            if best_gold_rel_id and best_score >= self.similarity_threshold:
                best_gold_rel = gold_relations_by_id[best_gold_rel_id]
                # Case-insensitive type matching
                type_matches = (not match_type) or (pred_rel.relation_type.upper() == best_gold_rel.type.upper())
                is_exact = best_score >= self.similarity_threshold and type_matches
                
                if is_exact:
                    true_positives.append(best_gold_rel)
                    matched_gold_relation_ids.add(best_gold_rel_id)
                else:
                    semantic_matches.append((pred_rel, best_gold_rel, best_score))
                    matched_gold_relation_ids.add(best_gold_rel_id)
            else:
                false_positives.append(pred_rel)
        
        false_negatives = [
            gold_rel for gold_rel in gold_relations_obj.relations
            if gold_rel.id not in matched_gold_relation_ids
        ]
        
        return true_positives, false_positives, false_negatives, semantic_matches

