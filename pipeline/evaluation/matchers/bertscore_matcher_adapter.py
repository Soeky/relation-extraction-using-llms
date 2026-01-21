"""BERTScore matcher adapter (wraps BERTScoreMatcher)."""

import logging
from typing import List, Tuple, Optional

try:
    from ..bertscore_matcher import BERTScoreMatcher
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    BERTScoreMatcher = None

from .base import BaseMatcher
from ...types import Relation, ParsedRelation, GoldRelations


class BERTScoreMatcherAdapter(BaseMatcher):
    """BERTScore matcher adapter wrapping BERTScoreMatcher."""
    
    def __init__(
        self,
        match_type: bool = True,
        similarity_threshold: float = 0.85,
        model_type: str = "microsoft/deberta-xlarge-mnli",
        use_openai_embeddings: bool = True,
        entity_map=None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize BERTScore matcher adapter.
        
        Args:
            match_type: Whether to require relation type to match (default: True)
            similarity_threshold: Minimum similarity for BERTScore matching (0-1)
            model_type: BERT model to use (or OpenAI embedding model if use_openai_embeddings=True)
            use_openai_embeddings: If True, use OpenAI embeddings API instead of BERTScore
            entity_map: Optional entity map for getting gold entity mentions
            logger: Optional logger instance
        """
        super().__init__(match_type, similarity_threshold, logger)
        self.model_type = model_type
        self.use_openai_embeddings = use_openai_embeddings
        self.entity_map = entity_map
        
        if not BERTSCORE_AVAILABLE or BERTScoreMatcher is None:
            raise ImportError(
                "BERTScoreMatcher not available. Install dependencies or ensure bertscore_matcher.py exists."
            )
        
        self._internal_matcher = BERTScoreMatcher(
            model_type=model_type,
            similarity_threshold=similarity_threshold,
            use_openai_embeddings=use_openai_embeddings,
            logger=logger
        )
    
    def get_name(self) -> str:
        """Get matcher name."""
        return "bertscore"
    
    def get_config(self) -> dict:
        """Get matcher configuration."""
        return {
            "match_type": self.match_type,
            "similarity_threshold": self.similarity_threshold,
            "model_type": self.model_type,
            "use_openai_embeddings": self.use_openai_embeddings,
            "method": "bertscore_semantic_similarity"
        }
    
    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute BERTScore similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        return self._internal_matcher.compute_entity_similarity(text1, text2)
    
    def match(
        self,
        predicted_relations: List[ParsedRelation],
        gold_relations_obj: GoldRelations,
        match_type: Optional[bool] = None
    ) -> Tuple[List[Relation], List[ParsedRelation], List[Relation], List[Tuple[ParsedRelation, Relation, float]]]:
        """
        Match predicted relations to gold relations using BERTScore.
        
        Args:
            predicted_relations: List of predicted relations
            gold_relations_obj: GoldRelations object
            match_type: Whether to require relation type to match (defaults to self.match_type)
            
        Returns:
            Tuple of (true_positives, false_positives, false_negatives, semantic_matches)
        """
        if match_type is None:
            match_type = self.match_type
        
        return self._internal_matcher.match_relations_with_bertscore(
            predicted_relations,
            gold_relations_obj.relations,
            entity_map=self.entity_map,
            use_exact_match_first=True,
            match_type=match_type
        )

