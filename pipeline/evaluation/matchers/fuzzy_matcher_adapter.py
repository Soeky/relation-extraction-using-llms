"""Fuzzy ID-based matcher adapter (wraps FuzzyRelationMatcher)."""

import logging
from typing import List, Tuple, Optional

from .base import BaseMatcher
from ..fuzzy_matcher import FuzzyRelationMatcher
from ...types import Relation, ParsedRelation, GoldRelations


class FuzzyMatcherAdapter(BaseMatcher):
    """Fuzzy ID-based matcher adapter wrapping FuzzyRelationMatcher."""
    
    def __init__(
        self,
        match_type: bool = True,
        similarity_threshold: float = 0.7,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize fuzzy matcher adapter.
        
        Args:
            match_type: Whether to require relation type to match (default: True)
            similarity_threshold: Minimum similarity for fuzzy matching (0-1)
            logger: Optional logger instance
        """
        super().__init__(match_type, similarity_threshold, logger)
        self._internal_matcher = FuzzyRelationMatcher(
            match_type=match_type,
            similarity_threshold=similarity_threshold
        )
    
    def get_name(self) -> str:
        """Get matcher name."""
        return "fuzzy"
    
    def get_config(self) -> dict:
        """Get matcher configuration."""
        return {
            "match_type": self.match_type,
            "similarity_threshold": self.similarity_threshold,
            "method": "fuzzy_id_matching"
        }
    
    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute fuzzy text similarity (uses internal matcher's similarity computation).
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # FuzzyRelationMatcher uses SequenceMatcher internally
        from difflib import SequenceMatcher
        norm1 = self._normalize_text(text1)
        norm2 = self._normalize_text(text2)
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def match(
        self,
        predicted_relations: List[ParsedRelation],
        gold_relations_obj: GoldRelations,
        match_type: Optional[bool] = None
    ) -> Tuple[List[Relation], List[ParsedRelation], List[Relation], List[Tuple[ParsedRelation, Relation, float]]]:
        """
        Match predicted relations to gold relations using fuzzy ID matching.
        
        Args:
            predicted_relations: List of predicted relations
            gold_relations_obj: GoldRelations object (note: fuzzy matcher works on IDs, not text)
            match_type: Whether to require relation type to match (defaults to self.match_type)
            
        Returns:
            Tuple of (true_positives, false_positives, false_negatives, semantic_matches)
            Note: converts partial_matches to semantic_matches with score 0.5
        """
        if match_type is None:
            match_type = self.match_type
        
        # Fuzzy matcher works on entity IDs, so we pass gold relations list directly
        true_positives, false_positives, false_negatives, partial_matches = \
            self._internal_matcher.match(predicted_relations, gold_relations_obj.relations)
        
        # Convert partial_matches to semantic_matches format
        semantic_matches = [(pred, gold, 0.5) for pred, gold in partial_matches]
        
        return true_positives, false_positives, false_negatives, semantic_matches

