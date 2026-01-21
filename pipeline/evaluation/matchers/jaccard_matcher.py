"""Jaccard similarity-based matcher for relation matching."""

import logging
from typing import List, Tuple, Optional, Set
import re

from .base import BaseMatcher
from ...types import Relation, ParsedRelation, GoldRelations


class JaccardMatcher(BaseMatcher):
    """Matches relations using Jaccard similarity on token sets."""
    
    def __init__(
        self,
        match_type: bool = True,
        similarity_threshold: float = 0.7,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Jaccard matcher.
        
        Args:
            match_type: Whether to require relation type to match (default: True)
            similarity_threshold: Minimum Jaccard similarity for a match (0-1)
            logger: Optional logger instance
        """
        super().__init__(match_type, similarity_threshold, logger)
    
    def get_name(self) -> str:
        """Get matcher name."""
        return "jaccard"
    
    def get_config(self) -> dict:
        """Get matcher configuration."""
        return {
            "match_type": self.match_type,
            "similarity_threshold": self.similarity_threshold,
            "method": "jaccard_token_set"
        }
    
    def _tokenize(self, text: str) -> Set[str]:
        """
        Tokenize text into word set.
        
        Args:
            text: Text to tokenize
            
        Returns:
            Set of tokens (words)
        """
        if not text:
            return set()
        normalized = self._normalize_text(text)
        # Split on whitespace and filter out empty strings
        tokens = [t for t in re.split(r'\s+', normalized) if t]
        return set(tokens)
    
    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute Jaccard similarity between two texts with containment checks.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Jaccard similarity score (0-1), boosted if containment detected
        """
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts for containment check
        norm1 = self._normalize_text(text1)
        norm2 = self._normalize_text(text2)
        
        # Exact match after normalization
        if norm1 == norm2:
            return 1.0
        
        tokens1 = self._tokenize(text1)
        tokens2 = self._tokenize(text2)
        
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        jaccard = intersection / union if union > 0 else 0.0
        
        # Check for containment (one set is subset of another) - boost similarity
        if tokens1.issubset(tokens2) or tokens2.issubset(tokens1):
            # If one is fully contained in the other, boost to at least 0.75
            jaccard = max(jaccard, 0.75)
        
        # Also check normalized text containment for phrase-level containment
        if norm1 in norm2 or norm2 in norm1:
            # Boost similarity for substring containment
            jaccard = max(jaccard, 0.80)
        
        return jaccard
    
    def _convert_gold_relation_to_text(
        self,
        gold_rel: Relation,
        gold_relations_obj: GoldRelations
    ) -> List[Tuple[str, str, str]]:
        """
        Convert gold relation to text-based representations.
        
        Args:
            gold_rel: Gold relation
            gold_relations_obj: GoldRelations object
            
        Returns:
            List of (head_mention, tail_mention, relation_type) tuples
        """
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
        """
        Match predicted relations to gold relations using Jaccard similarity.
        
        Args:
            predicted_relations: List of predicted relations
            gold_relations_obj: GoldRelations object
            match_type: Whether to require relation type to match (defaults to self.match_type)
            
        Returns:
            Tuple of (true_positives, false_positives, false_negatives, semantic_matches)
        """
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
                    # Compute head and tail similarities
                    head_sim = self.compute_text_similarity(pred_rel.head_mention, gold_head)
                    tail_sim = self.compute_text_similarity(pred_rel.tail_mention, gold_tail)
                    
                    # Check reverse direction
                    reverse_head_sim = self.compute_text_similarity(pred_rel.head_mention, gold_tail)
                    reverse_tail_sim = self.compute_text_similarity(pred_rel.tail_mention, gold_head)
                    
                    # Type match (case-insensitive)
                    type_match = 1.0 if (not match_type) or (pred_rel.relation_type.upper() == gold_type.upper()) else 0.0
                    
                    # Improved scoring: use weighted average but also consider max when entities are well-matched
                    # If both entities have decent similarity, use average; if one is high, favor it more
                    avg_forward = (head_sim + tail_sim) / 2.0
                    max_forward = max(head_sim, tail_sim)
                    # Use weighted combination: 60% average, 40% max (favors good matches)
                    forward_score = 0.6 * avg_forward + 0.4 * max_forward
                    if match_type:
                        forward_score = 0.35 * head_sim + 0.35 * tail_sim + 0.3 * type_match
                    
                    avg_reverse = (reverse_head_sim + reverse_tail_sim) / 2.0
                    max_reverse = max(reverse_head_sim, reverse_tail_sim)
                    reverse_score = 0.6 * avg_reverse + 0.4 * max_reverse
                    if match_type:
                        reverse_score = 0.35 * reverse_head_sim + 0.35 * reverse_tail_sim + 0.3 * type_match
                    
                    relation_score = max(forward_score, reverse_score)
                    best_relation_score = max(best_relation_score, relation_score)
                
                if best_relation_score > best_score:
                    best_score = best_relation_score
                    best_gold_rel_id = gold_rel_id
            
            # Check if we found a match
            if best_gold_rel_id and best_score >= self.similarity_threshold:
                best_gold_rel = gold_relations_by_id[best_gold_rel_id]
                
                # Check if type matches (if required, case-insensitive)
                type_matches = (not match_type) or (pred_rel.relation_type.upper() == best_gold_rel.type.upper())
                
                # Consider it a true positive if similarity is high enough and type matches (if required)
                is_exact = best_score >= self.similarity_threshold and type_matches
                
                if is_exact:
                    true_positives.append(best_gold_rel)
                    matched_gold_relation_ids.add(best_gold_rel_id)
                else:
                    # Semantic match
                    semantic_matches.append((pred_rel, best_gold_rel, best_score))
                    matched_gold_relation_ids.add(best_gold_rel_id)
            else:
                false_positives.append(pred_rel)
        
        # False negatives
        false_negatives = [
            gold_rel for gold_rel in gold_relations_obj.relations
            if gold_rel.id not in matched_gold_relation_ids
        ]
        
        return true_positives, false_positives, false_negatives, semantic_matches

