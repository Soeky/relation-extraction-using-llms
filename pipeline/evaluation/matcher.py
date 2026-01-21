"""Relation matcher for matching predictions to gold standard."""

from typing import List, Tuple, Set, Optional
from ..types import Relation, ParsedRelation


class RelationMatcher:
    """Matches predicted relations to gold standard relations."""
    
    def __init__(self, match_type: bool = True):
        """
        Initialize relation matcher.
        
        Args:
            match_type: Whether to require relation type to match (default: True)
        """
        self.match_type = match_type
    
    def match(
        self,
        predicted_relations: List[ParsedRelation],
        gold_relations: List[Relation]
    ) -> Tuple[List[Relation], List[ParsedRelation], List[Relation], List[Tuple[ParsedRelation, Relation]]]:
        """
        Match predicted relations to gold relations.
        
        Args:
            predicted_relations: List of predicted relations
            gold_relations: List of gold standard relations
            
        Returns:
            Tuple of (true_positives, false_positives, false_negatives, partial_matches)
            where partial_matches are (predicted, gold) pairs where entities match but type differs
        """
        # Convert gold relations to set of tuples for matching
        gold_set = self._relations_to_set(gold_relations)
        
        # Track matched gold relations by their tuple representation (hashable)
        matched_gold_tuples = set()
        true_positives = []
        false_positives = []
        partial_matches = []  # Entities match but type differs
        
        # Check each predicted relation
        for pred_rel in predicted_relations:
            if not pred_rel.head_id or not pred_rel.tail_id:
                # Cannot match without entity IDs
                false_positives.append(pred_rel)
                continue
            
            pred_tuple = self._relation_to_tuple(pred_rel)
            
            # Try to find matching gold relation
            matched = False
            partial_match_found = None
            
            for gold_rel in gold_relations:
                gold_tuple = self._relation_to_tuple_from_gold(gold_rel)
                
                # Check for exact match (including type if match_type=True)
                if self._tuples_match(pred_tuple, gold_tuple):
                    if gold_tuple not in matched_gold_tuples:
                        true_positives.append(gold_rel)
                        matched_gold_tuples.add(gold_tuple)
                        matched = True
                        break
                # Check for partial match (entities match but type differs)
                elif self.match_type and self._entities_match(pred_rel, gold_rel):
                    # Entities match but type doesn't - this is a partial match
                    partial_match_found = gold_rel
            
            if not matched:
                if partial_match_found:
                    partial_matches.append((pred_rel, partial_match_found))
                else:
                    false_positives.append(pred_rel)
        
        # False negatives: gold relations not matched
        false_negatives = [
            gold_rel for gold_rel in gold_relations
            if self._relation_to_tuple_from_gold(gold_rel) not in matched_gold_tuples
        ]
        
        return true_positives, false_positives, false_negatives, partial_matches
    
    def _entities_match(self, pred_rel: ParsedRelation, gold_rel: Relation) -> bool:
        """
        Check if entities match (ignoring relation type).
        
        Args:
            pred_rel: Predicted relation
            gold_rel: Gold relation
            
        Returns:
            True if head and tail entities match (in either direction)
        """
        # Check forward direction
        forward_match = (
            pred_rel.head_id == gold_rel.head_id and 
            pred_rel.tail_id == gold_rel.tail_id
        )
        
        # Check reverse direction
        reverse_match = (
            pred_rel.head_id == gold_rel.tail_id and 
            pred_rel.tail_id == gold_rel.head_id
        )
        
        return forward_match or reverse_match
    
    def _relation_to_tuple(self, relation: ParsedRelation) -> Tuple[str, str, Optional[str]]:
        """
        Convert ParsedRelation to tuple for matching.
        
        Args:
            relation: ParsedRelation object
            
        Returns:
            Tuple of (head_id, tail_id, relation_type)
        """
        rel_type = relation.relation_type if self.match_type else None
        return (relation.head_id, relation.tail_id, rel_type)
    
    def _relation_to_tuple_from_gold(self, relation: Relation) -> Tuple[str, str, Optional[str]]:
        """
        Convert Relation to tuple for matching.
        
        Args:
            relation: Relation object
            
        Returns:
            Tuple of (head_id, tail_id, relation_type)
        """
        rel_type = relation.type if self.match_type else None
        return (relation.head_id, relation.tail_id, rel_type)
    
    def _tuples_match(
        self, 
        tuple1: Tuple[str, str, Optional[str]], 
        tuple2: Tuple[str, str, Optional[str]]
    ) -> bool:
        """
        Check if two relation tuples match.
        
        Args:
            tuple1: First tuple
            tuple2: Second tuple
            
        Returns:
            True if tuples match
        """
        # Check both directions (head-tail and tail-head)
        forward_match = (
            tuple1[0] == tuple2[0] and 
            tuple1[1] == tuple2[1] and
            (tuple1[2] is None or tuple2[2] is None or tuple1[2] == tuple2[2])
        )
        
        reverse_match = (
            tuple1[0] == tuple2[1] and 
            tuple1[1] == tuple2[0] and
            (tuple1[2] is None or tuple2[2] is None or tuple1[2] == tuple2[2])
        )
        
        return forward_match or reverse_match
    
    def _relations_to_set(self, relations: List[Relation]) -> Set[Tuple[str, str, Optional[str]]]:
        """
        Convert list of relations to set of tuples.
        
        Args:
            relations: List of Relation objects
            
        Returns:
            Set of relation tuples
        """
        return {
            self._relation_to_tuple_from_gold(rel) for rel in relations
        }
