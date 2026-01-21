"""Exact ID-based matcher (wraps RelationMatcher)."""

import logging
from typing import List, Tuple, Optional

from .base import BaseMatcher
from ..matcher import RelationMatcher
from ...types import Relation, ParsedRelation, GoldRelations


class ExactMatcher(BaseMatcher):
    """Matches relations using exact ID matching (no variation allowed)."""
    
    def __init__(
        self,
        match_type: bool = True,
        similarity_threshold: float = 1.0,  # Exact matching, so threshold is 1.0
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize exact matcher.
        
        Args:
            match_type: Whether to require relation type to match (default: True)
            similarity_threshold: Not used for exact matching (kept for interface compatibility)
            logger: Optional logger instance
        """
        super().__init__(match_type, similarity_threshold, logger)
        self._internal_matcher = RelationMatcher(match_type=match_type)
    
    def get_name(self) -> str:
        """Get matcher name."""
        return "exact"
    
    def get_config(self) -> dict:
        """Get matcher configuration."""
        return {
            "match_type": self.match_type,
            "similarity_threshold": self.similarity_threshold,
            "method": "exact_id_matching"
        }
    
    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute exact text similarity (1.0 if equal, 0.0 otherwise).
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            1.0 if texts are equal after normalization, 0.0 otherwise
        """
        norm1 = self._normalize_text(text1)
        norm2 = self._normalize_text(text2)
        return 1.0 if norm1 == norm2 else 0.0
    
    def match(
        self,
        predicted_relations: List[ParsedRelation],
        gold_relations_obj: GoldRelations,
        match_type: Optional[bool] = None
    ) -> Tuple[List[Relation], List[ParsedRelation], List[Relation], List[Tuple[ParsedRelation, Relation, float]]]:
        """
        Match predicted relations to gold relations using exact ID matching.
        
        Args:
            predicted_relations: List of predicted relations
            gold_relations_obj: GoldRelations object
            match_type: Whether to require relation type to match (defaults to self.match_type)
            
        Returns:
            Tuple of (true_positives, false_positives, false_negatives, semantic_matches)
            Note: semantic_matches will be empty for exact matching, but we convert partial_matches
        """
        if match_type is None:
            match_type = self.match_type
        
        # Use internal matcher
        true_positives, false_positives, false_negatives, partial_matches = \
            self._internal_matcher.match(predicted_relations, gold_relations_obj.relations)
        
        # Convert partial_matches to semantic_matches format (with score 0.5 to indicate partial match)
        semantic_matches = [(pred, gold, 0.5) for pred, gold in partial_matches]
        
        return true_positives, false_positives, false_negatives, semantic_matches

