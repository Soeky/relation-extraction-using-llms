"""Entity resolver for mapping mentions to IDs."""

import re
import string
from typing import List, Optional, Tuple, Set
from difflib import SequenceMatcher

from ..types import ParsedRelation, GlobalEntity
from ..data.entity_map import GlobalEntityMap


class EntityResolver:
    """Resolves entity mentions to global entity IDs."""
    
    def __init__(self, entity_map: Optional[GlobalEntityMap] = None, fuzzy_threshold: float = 0.7):
        """
        Initialize entity resolver.
        
        Args:
            entity_map: Global entity map for resolution
            fuzzy_threshold: Minimum similarity threshold for fuzzy matching (0-1)
        """
        self.entity_map = entity_map
        self.fuzzy_threshold = fuzzy_threshold
    
    def resolve_mention(
        self, 
        mention_text: str, 
        entity_type: Optional[str] = None,
        source_text: Optional[str] = None,
        document_entity_ids: Optional[Set[str]] = None
    ) -> Optional[str]:
        """
        Resolve a mention to an entity ID with improved fuzzy matching.
        
        Args:
            mention_text: Text mention to resolve
            entity_type: Optional entity type hint
            source_text: Optional source text for context
            document_entity_ids: Optional set of entity IDs to limit search to (entities in current document)
            
        Returns:
            Entity ID or None if not found
        """
        if not self.entity_map:
            return None
        
        mention_text = mention_text.strip()
        if not mention_text:
            return None
        
        # Filter entities to only those in current document if provided
        entities_to_search = self.entity_map.entities
        if document_entity_ids:
            entities_to_search = {
                eid: entity for eid, entity in self.entity_map.entities.items()
                if eid in document_entity_ids
            }
            if not entities_to_search:
                # If filtering results in empty set, fall back to all entities
                # (this shouldn't happen, but safety check)
                entities_to_search = self.entity_map.entities
        
        # Normalize mention text
        normalized_mention = self._normalize_text(mention_text)
        
        # Try exact match first (case-insensitive) - filter by document entities if provided
        matches = self.entity_map.find_entity_by_mention(
            mention_text, 
            entity_type=entity_type, 
            fuzzy=False
        )
        
        # Filter matches to document entities if provided
        if matches:
            if document_entity_ids:
                matches = [m for m in matches if m.id in document_entity_ids]
            if matches:
                return matches[0].id
        
        # Try normalized exact match
        for entity_id, entity in entities_to_search.items():
            if entity_type and entity.type != entity_type:
                continue
            
            normalized_canonical = self._normalize_text(entity.canonical_name)
            if normalized_mention == normalized_canonical:
                return entity.id
            
            # Check common mentions
            for common_mention in entity.common_mentions:
                if normalized_mention == self._normalize_text(common_mention):
                    return entity.id
        
        # Try fuzzy match with similarity scoring
        best_match = None
        best_score = 0.0
        
        for entity_id, entity in entities_to_search.items():
            if entity_type and entity.type != entity_type:
                continue
            
            score = self._similarity_score(mention_text, entity)
            if score > best_score and score >= self.fuzzy_threshold:
                best_score = score
                best_match = entity
        
        if best_match:
            return best_match.id
        
        # Try partial matching for long entity descriptions
        # This helps with cases like "15 nucleotide deletion" vs "deletion of the coding sequence (nt 1314 through nt 1328)"
        if len(mention_text) > 10:  # Only for longer mentions
            best_match = self._partial_match(mention_text, entity_type, document_entity_ids)
            if best_match:
                return best_match.id
        
        return None
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison (lowercase, remove punctuation, normalize whitespace).
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Lowercase
        normalized = text.lower()
        
        # Remove punctuation but keep spaces
        normalized = normalized.translate(str.maketrans('', '', string.punctuation))
        
        # Normalize whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def _extract_core_terms(self, text: str) -> set:
        """
        Extract core terms from text (remove common words, keep important terms).
        
        Args:
            text: Text to extract terms from
            
        Returns:
            Set of core terms
        """
        # Common stop words in biomedical context
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        normalized = self._normalize_text(text)
        words = normalized.split()
        core_terms = {w for w in words if w not in stop_words and len(w) > 2}
        return core_terms
    
    def _similarity_score(self, mention_text: str, entity: GlobalEntity) -> float:
        """
        Calculate improved similarity score between mention and entity.
        
        Uses multiple strategies:
        1. Sequence matcher on full text
        2. Core term overlap
        3. Partial substring matching
        
        Args:
            mention_text: Mention text
            entity: GlobalEntity to compare
            
        Returns:
            Similarity score (0-1)
        """
        mention_lower = mention_text.lower().strip()
        mention_normalized = self._normalize_text(mention_text)
        mention_terms = self._extract_core_terms(mention_text)
        
        best_score = 0.0
        
        # Check against canonical name
        if entity.canonical_name:
            canon_lower = entity.canonical_name.lower()
            canon_normalized = self._normalize_text(entity.canonical_name)
            canon_terms = self._extract_core_terms(entity.canonical_name)
            
            # Sequence matcher
            seq_score = SequenceMatcher(None, mention_normalized, canon_normalized).ratio()
            best_score = max(best_score, seq_score)
            
            # Core term overlap (Jaccard similarity)
            if mention_terms and canon_terms:
                overlap = len(mention_terms & canon_terms)
                union = len(mention_terms | canon_terms)
                term_score = overlap / union if union > 0 else 0.0
                best_score = max(best_score, term_score)
            
            # Partial match bonus (if one contains the other)
            if mention_normalized in canon_normalized or canon_normalized in mention_normalized:
                best_score = max(best_score, 0.85)  # High score for containment
        
        # Check against common mentions
        for common_mention in entity.common_mentions[:10]:  # Check top 10
            cm_lower = common_mention.lower()
            cm_normalized = self._normalize_text(common_mention)
            cm_terms = self._extract_core_terms(common_mention)
            
            # Sequence matcher
            seq_score = SequenceMatcher(None, mention_normalized, cm_normalized).ratio()
            best_score = max(best_score, seq_score)
            
            # Core term overlap
            if mention_terms and cm_terms:
                overlap = len(mention_terms & cm_terms)
                union = len(mention_terms | cm_terms)
                term_score = overlap / union if union > 0 else 0.0
                best_score = max(best_score, term_score)
            
            # Partial match bonus
            if mention_normalized in cm_normalized or cm_normalized in mention_normalized:
                best_score = max(best_score, 0.85)
        
        return best_score
    
    def _partial_match(
        self, 
        mention_text: str, 
        entity_type: Optional[str] = None,
        document_entity_ids: Optional[Set[str]] = None
    ) -> Optional[GlobalEntity]:
        """
        Try to find partial matches for long entity descriptions.
        
        This helps with cases where the LLM extracts a shorter mention
        but the gold standard has a longer, more complete description.
        
        Args:
            mention_text: Mention text to match
            entity_type: Optional entity type filter
            document_entity_ids: Optional set of entity IDs to limit search to
            
        Returns:
            Best matching GlobalEntity or None
        """
        mention_normalized = self._normalize_text(mention_text)
        mention_terms = self._extract_core_terms(mention_text)
        
        if not mention_terms:
            return None
        
        # Filter entities to only those in current document if provided
        entities_to_search = self.entity_map.entities
        if document_entity_ids:
            entities_to_search = {
                eid: entity for eid, entity in self.entity_map.entities.items()
                if eid in document_entity_ids
            }
        
        best_match = None
        best_score = 0.0
        
        for entity_id, entity in entities_to_search.items():
            if entity_type and entity.type != entity_type:
                continue
            
            # Check if mention terms appear in entity mentions
            for mention in entity.all_mentions[:20]:  # Check first 20 mentions
                mention_obj_normalized = self._normalize_text(mention.text)
                mention_obj_terms = self._extract_core_terms(mention.text)
                
                # Check if most core terms from mention appear in entity mention
                if mention_obj_terms:
                    overlap = len(mention_terms & mention_obj_terms)
                    coverage = overlap / len(mention_terms) if mention_terms else 0.0
                    
                    # Require at least 60% of core terms to match
                    if coverage >= 0.6:
                        # Calculate final score
                        score = coverage * 0.7 + SequenceMatcher(
                            None, mention_normalized, mention_obj_normalized
                        ).ratio() * 0.3
                        
                        if score > best_score:
                            best_score = score
                            best_match = entity
        
        # Only return if score is above threshold
        if best_score >= self.fuzzy_threshold:
            return best_match
        
        return None
    
    def resolve_relation(
        self, 
        relation: ParsedRelation,
        source_text: Optional[str] = None,
        document_entity_ids: Optional[Set[str]] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Resolve both head and tail entities in a relation.
        
        Args:
            relation: ParsedRelation to resolve
            source_text: Optional source text for context
            document_entity_ids: Optional set of entity IDs to limit search to
            
        Returns:
            Tuple of (head_id, tail_id)
        """
        head_id = self.resolve_mention(
            relation.head_mention,
            source_text=source_text,
            document_entity_ids=document_entity_ids
        )
        
        tail_id = self.resolve_mention(
            relation.tail_mention,
            source_text=source_text,
            document_entity_ids=document_entity_ids
        )
        
        return head_id, tail_id
    
    def resolve_relations(
        self,
        relations: List[ParsedRelation],
        source_text: Optional[str] = None,
        document_entity_ids: Optional[Set[str]] = None
    ) -> List[ParsedRelation]:
        """
        Resolve all relations in a list.
        
        Args:
            relations: List of ParsedRelation objects
            source_text: Optional source text for context
            document_entity_ids: Optional set of entity IDs to limit search to (entities in current document)
            
        Returns:
            List of ParsedRelation objects with resolved IDs
        """
        resolved = []
        for relation in relations:
            head_id, tail_id = self.resolve_relation(
                relation, 
                source_text=source_text,
                document_entity_ids=document_entity_ids
            )
            relation.head_id = head_id
            relation.tail_id = tail_id
            resolved.append(relation)
        
        return resolved
