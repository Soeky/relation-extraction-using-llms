"""Post-processing validator for relations."""

import re
import logging
from typing import List, Optional, Tuple

from ..types import ParsedRelation, ParsedRelations


class RelationValidator:
    """Validates extracted relations against source text."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize relation validator.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def validate_relation(
        self, 
        relation: ParsedRelation, 
        source_text: str,
        strict: bool = False
    ) -> Tuple[bool, str]:
        """
        Validate a single relation against source text.
        
        Args:
            relation: ParsedRelation to validate
            source_text: Source document text
            strict: If True, require exact matches; if False, allow fuzzy matches
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check if head mention appears in source text
        head_found = self._find_mention_in_text(relation.head_mention, source_text, strict)
        if not head_found:
            return False, f"Head entity '{relation.head_mention}' not found in source text"
        
        # Check if tail mention appears in source text
        tail_found = self._find_mention_in_text(relation.tail_mention, source_text, strict)
        if not tail_found:
            return False, f"Tail entity '{relation.tail_mention}' not found in source text"
        
        # Check if relation type is valid
        valid_types = [
            "Association", "Positive_Correlation", "Negative_Correlation",
            "Causal", "Part_Of", "Regulates", "Treats", "Causes",
            "Prevents", "Increases", "Decreases", "Coexists_With"
        ]
        if relation.relation_type not in valid_types:
            # Allow but warn about unknown types
            self.logger.warning(
                f"Unknown relation type: {relation.relation_type} "
                f"(head: {relation.head_mention}, tail: {relation.tail_mention})"
            )
        
        return True, "Valid"
    
    def _find_mention_in_text(
        self, 
        mention: str, 
        text: str, 
        strict: bool = False
    ) -> bool:
        """
        Check if mention appears in text.
        
        Args:
            mention: Entity mention to find
            text: Source text
            strict: If True, require exact match; if False, allow case-insensitive and partial
            
        Returns:
            True if found, False otherwise
        """
        if not mention or not text:
            return False
        
        mention = mention.strip()
        if not mention:
            return False
        
        if strict:
            # Exact match
            return mention in text
        else:
            # Case-insensitive match
            mention_lower = mention.lower()
            text_lower = text.lower()
            
            # Try exact match first
            if mention_lower in text_lower:
                return True
            
            # Try normalized match (remove punctuation, normalize whitespace)
            mention_normalized = re.sub(r'[^\w\s]', '', mention_lower)
            mention_normalized = ' '.join(mention_normalized.split())
            
            text_normalized = re.sub(r'[^\w\s]', '', text_lower)
            text_normalized = ' '.join(text_normalized.split())
            
            if mention_normalized in text_normalized:
                return True
            
            # Try partial match for long mentions (check if key terms appear)
            if len(mention.split()) > 3:
                mention_words = set(mention_normalized.split())
                text_words = set(text_normalized.split())
                
                # Check if at least 60% of mention words appear in text
                overlap = mention_words & text_words
                if len(mention_words) > 0 and len(overlap) / len(mention_words) >= 0.6:
                    return True
        
        return False
    
    def validate_relations(
        self, 
        relations: List[ParsedRelation],
        source_text: str,
        strict: bool = False,
        filter_invalid: bool = True
    ) -> Tuple[List[ParsedRelation], List[str]]:
        """
        Validate multiple relations and optionally filter invalid ones.
        
        Args:
            relations: List of ParsedRelation objects
            source_text: Source document text
            strict: If True, require exact matches
            filter_invalid: If True, remove invalid relations from result
            
        Returns:
            Tuple of (valid_relations, validation_errors)
        """
        valid_relations = []
        validation_errors = []
        
        for relation in relations:
            is_valid, reason = self.validate_relation(relation, source_text, strict)
            
            if is_valid:
                valid_relations.append(relation)
            else:
                validation_errors.append(
                    f"Invalid relation: {relation.head_mention} -> "
                    f"{relation.tail_mention} ({relation.relation_type}): {reason}"
                )
                if not filter_invalid:
                    # Still include but mark as invalid
                    valid_relations.append(relation)
        
        if validation_errors:
            self.logger.warning(
                f"Found {len(validation_errors)} invalid relations out of {len(relations)}"
            )
            for error in validation_errors[:5]:  # Log first 5 errors
                self.logger.debug(f"[Validator] {error}")
        
        return valid_relations, validation_errors
    
    def assign_confidence_scores(
        self,
        relations: List[ParsedRelation],
        source_text: str
    ) -> List[ParsedRelation]:
        """
        Assign confidence scores to relations based on validation.
        
        Args:
            relations: List of ParsedRelation objects
            source_text: Source document text
            
        Returns:
            List of ParsedRelation objects with confidence scores
        """
        for relation in relations:
            # Base confidence
            confidence = 0.5
            
            # Boost confidence if mentions found exactly
            head_found_strict = self._find_mention_in_text(relation.head_mention, source_text, strict=True)
            tail_found_strict = self._find_mention_in_text(relation.tail_mention, source_text, strict=True)
            
            if head_found_strict:
                confidence += 0.2
            if tail_found_strict:
                confidence += 0.2
            
            # Boost if both found (even if not strict)
            head_found = self._find_mention_in_text(relation.head_mention, source_text, strict=False)
            tail_found = self._find_mention_in_text(relation.tail_mention, source_text, strict=False)
            
            if head_found and tail_found:
                confidence += 0.1
            
            # Cap at 1.0
            relation.confidence = min(1.0, confidence)
        
        return relations

