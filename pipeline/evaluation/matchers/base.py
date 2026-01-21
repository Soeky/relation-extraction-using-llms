"""Base matcher interface for relation matching."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import logging
import re

from ...types import Relation, ParsedRelation, GoldRelations


class BaseMatcher(ABC):
    """Abstract base class for relation matchers."""
    
    def __init__(
        self,
        match_type: bool = True,
        similarity_threshold: float = 0.7,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize base matcher.
        
        Args:
            match_type: Whether to require relation type to match (default: True)
            similarity_threshold: Minimum similarity for a match (0-1)
            logger: Optional logger instance
        """
        self.match_type = match_type
        self.similarity_threshold = similarity_threshold
        self.logger = logger or logging.getLogger(__name__)
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of this matcher strategy.
        
        Returns:
            String name of the matcher (e.g., "jaccard", "text", "bertscore")
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration dictionary for this matcher.
        
        Returns:
            Dictionary with matcher configuration parameters
        """
        pass
    
    @abstractmethod
    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        pass
    
    @abstractmethod
    def match(
        self,
        predicted_relations: List[ParsedRelation],
        gold_relations_obj: GoldRelations,
        match_type: Optional[bool] = None
    ) -> Tuple[List[Relation], List[ParsedRelation], List[Relation], List[Tuple[ParsedRelation, Relation, float]]]:
        """
        Match predicted relations to gold relations.
        
        Args:
            predicted_relations: List of predicted relations
            gold_relations_obj: GoldRelations object containing gold relations and entities
            match_type: Whether to require relation type to match (defaults to self.match_type)
            
        Returns:
            Tuple of (true_positives, false_positives, false_negatives, semantic_matches)
            where semantic_matches are (predicted, gold, similarity_score) tuples
        """
        pass
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for matching (enhanced implementation with whitespace normalization).
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Lowercase and strip
        normalized = text.lower().strip()
        
        # Remove extra whitespace (like text matcher does)
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized

