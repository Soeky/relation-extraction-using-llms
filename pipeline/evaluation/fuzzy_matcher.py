"""Fuzzy relation matcher that uses string similarity for entity IDs."""

from typing import List, Tuple, Optional
from difflib import SequenceMatcher
from ..types import Relation, ParsedRelation


class FuzzyRelationMatcher:
    """Matches predicted relations to gold standard using fuzzy string matching on entity IDs."""
    
    def __init__(self, match_type: bool = True, similarity_threshold: float = 0.7):
        """
        Initialize fuzzy relation matcher.
        
        Args:
            match_type: Whether to require relation type to match (default: True)
            similarity_threshold: Minimum similarity for fuzzy matching (0-1, default: 0.7)
        """
        self.match_type = match_type
        self.similarity_threshold = similarity_threshold
    
    def match(
        self,
        predicted_relations: List[ParsedRelation],
        gold_relations: List[Relation]
    ) -> Tuple[List[Relation], List[ParsedRelation], List[Relation], List[Tuple[ParsedRelation, Relation]]]:
        """
        Match predicted relations to gold relations using fuzzy string matching on entity IDs.
        
        Args:
            predicted_relations: List of predicted relations
            gold_relations: List of gold standard relations
            
        Returns:
            Tuple of (true_positives, false_positives, false_negatives, partial_matches)
            where partial_matches are (predicted, gold) pairs where entities match but type differs
        """
        # Track matched gold relations
        matched_gold_indices = set()
        true_positives = []
        false_positives = []
        partial_matches = []  # Entities match but type differs
        
        # Check each predicted relation
        for pred_rel in predicted_relations:
            if not pred_rel.head_id or not pred_rel.tail_id:
                # Cannot match without entity IDs
                false_positives.append(pred_rel)
                continue
            
            # Try to find matching gold relation
            matched = False
            partial_match_found = None
            best_match_idx = None
            best_similarity = 0.0
            
            for idx, gold_rel in enumerate(gold_relations):
                if idx in matched_gold_indices:
                    continue
                
                # Compute entity ID similarities
                head_sim = self._compute_similarity(pred_rel.head_id, gold_rel.head_id)
                tail_sim = self._compute_similarity(pred_rel.tail_id, gold_rel.tail_id)
                
                # Check forward direction
                forward_match = (
                    head_sim >= self.similarity_threshold and 
                    tail_sim >= self.similarity_threshold
                )
                
                # Check reverse direction
                reverse_head_sim = self._compute_similarity(pred_rel.head_id, gold_rel.tail_id)
                reverse_tail_sim = self._compute_similarity(pred_rel.tail_id, gold_rel.head_id)
                reverse_match = (
                    reverse_head_sim >= self.similarity_threshold and 
                    reverse_tail_sim >= self.similarity_threshold
                )
                
                # Check if entities match (either direction)
                entities_match = forward_match or reverse_match
                
                if entities_match:
                    # Check relation type
                    type_matches = (
                        not self.match_type or 
                        pred_rel.relation_type == gold_rel.type
                    )
                    
                    # Compute overall similarity (average of entity similarities)
                    if forward_match:
                        similarity = (head_sim + tail_sim) / 2.0
                    else:
                        similarity = (reverse_head_sim + reverse_tail_sim) / 2.0
                    
                    if type_matches:
                        # Full match (entities + type)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match_idx = idx
                            matched = True
                    else:
                        # Partial match (entities match but type differs)
                        if not partial_match_found or similarity > best_similarity:
                            partial_match_found = gold_rel
                            best_similarity = similarity
            
            if matched and best_match_idx is not None:
                # Add as true positive
                true_positives.append(gold_relations[best_match_idx])
                matched_gold_indices.add(best_match_idx)
            elif partial_match_found:
                # Add as partial match
                partial_matches.append((pred_rel, partial_match_found))
            else:
                # No match found
                false_positives.append(pred_rel)
        
        # False negatives: gold relations not matched
        false_negatives = [
            gold_rel for idx, gold_rel in enumerate(gold_relations)
            if idx not in matched_gold_indices
        ]
        
        return true_positives, false_positives, false_negatives, partial_matches
    
    def _compute_similarity(self, id1: str, id2: str) -> float:
        """
        Compute string similarity between two entity IDs.
        
        Args:
            id1: First entity ID
            id2: Second entity ID
            
        Returns:
            Similarity score (0-1)
        """
        if not id1 or not id2:
            return 0.0
        
        # Normalize IDs (lowercase, strip whitespace)
        norm1 = id1.lower().strip()
        norm2 = id2.lower().strip()
        
        # Exact match after normalization
        if norm1 == norm2:
            return 1.0
        
        # Use SequenceMatcher for fuzzy matching
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        
        return similarity

