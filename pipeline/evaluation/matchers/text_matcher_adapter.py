"""Text-based matcher adapter (wraps TextRelationMatcher)."""

import logging
from typing import List, Tuple, Optional

from .base import BaseMatcher
from ..text_matcher import TextRelationMatcher
from ...types import Relation, ParsedRelation, GoldRelations


class TextMatcherAdapter(BaseMatcher):
    """Text-based matcher adapter wrapping TextRelationMatcher."""
    
    def __init__(
        self,
        match_type: bool = True,
        similarity_threshold: float = 0.85,
        fuzzy_threshold: float = 0.7,
        use_bertscore: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize text matcher adapter.
        
        Args:
            match_type: Whether to require relation type to match (default: True)
            similarity_threshold: Minimum similarity for BERTScore matching (0-1)
            fuzzy_threshold: Minimum similarity for fuzzy string matching (0-1)
            use_bertscore: Whether to use BERTScore for semantic similarity
            logger: Optional logger instance
        """
        super().__init__(match_type, similarity_threshold, logger)
        self.use_bertscore = use_bertscore
        self.fuzzy_threshold = fuzzy_threshold
        self._internal_matcher = TextRelationMatcher(
            use_bertscore=use_bertscore,
            similarity_threshold=similarity_threshold,
            fuzzy_threshold=fuzzy_threshold,
            logger=logger
        )
    
    def get_name(self) -> str:
        """Get matcher name."""
        return "text"
    
    def get_config(self) -> dict:
        """Get matcher configuration."""
        return {
            "match_type": self.match_type,
            "similarity_threshold": self.similarity_threshold,
            "fuzzy_threshold": self.fuzzy_threshold,
            "use_bertscore": self.use_bertscore,
            "method": "text_based_matching"
        }
    
    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute text similarity using internal matcher.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        return self._internal_matcher.compute_text_similarity(text1, text2)
    
    def match(
        self,
        predicted_relations: List[ParsedRelation],
        gold_relations_obj: GoldRelations,
        match_type: Optional[bool] = None
    ) -> Tuple[List[Relation], List[ParsedRelation], List[Relation], List[Tuple[ParsedRelation, Relation, float]]]:
        """
        Match predicted relations to gold relations using text-based matching.
        
        Args:
            predicted_relations: List of predicted relations
            gold_relations_obj: GoldRelations object
            match_type: Whether to require relation type to match (defaults to self.match_type)
            
        Returns:
            Tuple of (true_positives, false_positives, false_negatives, semantic_matches)
        """
        if match_type is None:
            match_type = self.match_type
        
        return self._internal_matcher.match(predicted_relations, gold_relations_obj, match_type=match_type)

