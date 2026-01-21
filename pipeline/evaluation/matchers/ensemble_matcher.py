"""Ensemble matcher combining multiple matching strategies."""

import logging
from typing import List, Tuple, Optional, Dict

from .base import BaseMatcher
from ...types import Relation, ParsedRelation, GoldRelations


class EnsembleMatcher(BaseMatcher):
    """Matches relations using a weighted ensemble of multiple matchers."""
    
    def __init__(
        self,
        matchers: List[BaseMatcher],
        weights: Optional[List[float]] = None,
        match_type: bool = True,
        similarity_threshold: float = 0.7,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize ensemble matcher.
        
        Args:
            matchers: List of BaseMatcher instances to combine
            weights: Optional weights for each matcher (defaults to equal weights)
            match_type: Whether to require relation type to match (default: True)
            similarity_threshold: Minimum combined similarity for a match (0-1)
            logger: Optional logger instance
        """
        super().__init__(match_type, similarity_threshold, logger)
        
        if not matchers:
            raise ValueError("EnsembleMatcher requires at least one matcher")
        
        self.matchers = matchers
        
        if weights is None:
            # Equal weights
            weights = [1.0 / len(matchers)] * len(matchers)
        
        if len(weights) != len(matchers):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of matchers ({len(matchers)})")
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("Weights cannot all be zero")
        
        self.weights = [w / total_weight for w in weights]
    
    def get_name(self) -> str:
        """Get matcher name."""
        matcher_names = [m.get_name() for m in self.matchers]
        return f"ensemble_{'+'.join(matcher_names)}"
    
    def get_config(self) -> dict:
        """Get matcher configuration."""
        return {
            "match_type": self.match_type,
            "similarity_threshold": self.similarity_threshold,
            "matchers": [m.get_name() for m in self.matchers],
            "weights": self.weights,
            "method": "weighted_ensemble"
        }
    
    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute ensemble similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Weighted average similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        similarities = []
        for matcher, weight in zip(self.matchers, self.weights):
            try:
                sim = matcher.compute_text_similarity(text1, text2)
                similarities.append(sim * weight)
            except Exception as e:
                self.logger.warning(
                    f"[EnsembleMatcher] Error computing similarity with {matcher.get_name()}: {e}"
                )
        
        if not similarities:
            return 0.0
        
        return sum(similarities)
    
    def match(
        self,
        predicted_relations: List[ParsedRelation],
        gold_relations_obj: GoldRelations,
        match_type: Optional[bool] = None
    ) -> Tuple[List[Relation], List[ParsedRelation], List[Relation], List[Tuple[ParsedRelation, Relation, float]]]:
        """
        Match predicted relations to gold relations using ensemble of matchers.
        
        This computes similarities using all matchers and combines them.
        The actual matching logic uses the combined similarity scores.
        """
        if match_type is None:
            match_type = self.match_type
        
        # Use the first matcher's text conversion logic (they should all be similar)
        # But compute similarities using ensemble
        primary_matcher = self.matchers[0]
        
        # Convert gold relations to text (using primary matcher's method)
        gold_text_relations_map = {}
        gold_relations_by_id = {}
        
        # Get conversion method from primary matcher
        convert_method = getattr(primary_matcher, '_convert_gold_relation_to_text', None)
        if not convert_method:
            # Fallback: try to use a standard conversion
            for gold_rel in gold_relations_obj.relations:
                head_entity = None
                tail_entity = None
                for entity in gold_relations_obj.entities:
                    if entity.id == gold_rel.head_id:
                        head_entity = entity
                    if entity.id == gold_rel.tail_id:
                        tail_entity = entity
                
                if head_entity and tail_entity:
                    head_mentions = [m.text for m in head_entity.mentions] if head_entity.mentions else [gold_rel.head_id]
                    tail_mentions = [m.text for m in tail_entity.mentions] if tail_entity.mentions else [gold_rel.tail_id]
                    text_relations = [(h, t, gold_rel.type) for h in head_mentions for t in tail_mentions]
                    if text_relations:
                        gold_text_relations_map[gold_rel.id] = text_relations
                        gold_relations_by_id[gold_rel.id] = gold_rel
        else:
            for gold_rel in gold_relations_obj.relations:
                text_relations = convert_method(gold_rel, gold_relations_obj)
                if text_relations:
                    gold_text_relations_map[gold_rel.id] = text_relations
                    gold_relations_by_id[gold_rel.id] = gold_rel
        
        matched_gold_relation_ids = set()
        true_positives = []
        false_positives = []
        semantic_matches = []
        
        for pred_rel in predicted_relations:
            best_gold_rel_id = None
            best_score = 0.0
            
            for gold_rel_id, gold_text_relations in gold_text_relations_map.items():
                if gold_rel_id in matched_gold_relation_ids:
                    continue
                
                best_relation_score = 0.0
                
                for gold_head, gold_tail, gold_type in gold_text_relations:
                    # Compute ensemble similarity for head and tail
                    head_sim = self.compute_text_similarity(pred_rel.head_mention, gold_head)
                    tail_sim = self.compute_text_similarity(pred_rel.tail_mention, gold_tail)
                    reverse_head_sim = self.compute_text_similarity(pred_rel.head_mention, gold_tail)
                    reverse_tail_sim = self.compute_text_similarity(pred_rel.tail_mention, gold_head)
                    
                    # Type match (case-insensitive)
                    type_match = 1.0 if (not match_type) or (pred_rel.relation_type.upper() == gold_type.upper()) else 0.0
                    
                    forward_score = 0.4 * head_sim + 0.4 * tail_sim + (0.2 * type_match if match_type else 0.0)
                    reverse_score = 0.4 * reverse_head_sim + 0.4 * reverse_tail_sim + (0.2 * type_match if match_type else 0.0)
                    
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

