"""Sentence Transformers (SBERT) based matcher for relation matching."""

import logging
from typing import List, Tuple, Optional, Dict
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    SentenceTransformer = None

from .base import BaseMatcher
from ...types import Relation, ParsedRelation, GoldRelations


class SBERTMatcher(BaseMatcher):
    """Matches relations using Sentence Transformers (SBERT) embeddings."""
    
    def __init__(
        self,
        match_type: bool = True,
        similarity_threshold: float = 0.7,
        model_name: str = "all-MiniLM-L6-v2",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize SBERT matcher.
        
        Args:
            match_type: Whether to require relation type to match (default: True)
            similarity_threshold: Minimum cosine similarity for a match (0-1)
            model_name: Name of the SBERT model to use
            logger: Optional logger instance
        """
        super().__init__(match_type, similarity_threshold, logger)
        self.model_name = model_name
        self.model = None
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
        if SBERT_AVAILABLE:
            try:
                self.logger.info(f"[SBERTMatcher] Loading model: {model_name}")
                self.model = SentenceTransformer(model_name)
                self.logger.info(f"[SBERTMatcher] Model loaded successfully")
            except Exception as e:
                self.logger.warning(
                    f"[SBERTMatcher] Failed to load model {model_name}: {e}. "
                    "SBERT matching will not be available."
                )
                self.model = None
        else:
            self.logger.warning(
                "[SBERTMatcher] sentence-transformers not available. "
                "Install with: pip install sentence-transformers. "
                "Falling back to basic string matching."
            )
    
    def get_name(self) -> str:
        """Get matcher name."""
        return "sbert"
    
    def get_config(self) -> dict:
        """Get matcher configuration."""
        return {
            "match_type": self.match_type,
            "similarity_threshold": self.similarity_threshold,
            "model_name": self.model_name,
            "method": "sbert_cosine_similarity"
        }
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for text (with caching).
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if model not available
        """
        if not self.model or not text:
            return None
        
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        try:
            embedding = self.model.encode([text], convert_to_numpy=True)[0]
            self._embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            self.logger.warning(f"[SBERTMatcher] Error encoding text '{text[:50]}...': {e}")
            return None
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))
    
    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute SBERT-based similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score (0-1, typically -1 to 1 but embeddings are usually 0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        if not self.model:
            # Fallback
            norm1 = self._normalize_text(text1)
            norm2 = self._normalize_text(text2)
            return 1.0 if norm1 == norm2 else 0.0
        
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        similarity = self._cosine_similarity(emb1, emb2)
        # Normalize to 0-1 range (cosine similarity is -1 to 1, but embeddings typically yield 0-1)
        return max(0.0, similarity)
    
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
        """Match predicted relations to gold relations using SBERT embeddings."""
        if match_type is None:
            match_type = self.match_type
        
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

