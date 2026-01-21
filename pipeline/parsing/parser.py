"""Response parser for LLM outputs."""

import json
import re
import logging
from typing import List, Optional, Set

from ..types import ParsedRelations, ParsedRelation
from .entity_resolver import EntityResolver
from .validator import RelationValidator


class ResponseParser:
    """Parses LLM text responses into structured relations."""
    
    def __init__(self, entity_map=None, logger: Optional[logging.Logger] = None, validate: bool = True):
        """
        Initialize response parser.
        
        Args:
            entity_map: Optional global entity map for entity resolution
            logger: Optional logger instance
            validate: Whether to validate relations against source text
        """
        self.entity_resolver = EntityResolver(entity_map) if entity_map else None
        self.validator = RelationValidator(logger) if validate else None
        self.logger = logger or logging.getLogger(__name__)
        self.validate = validate
    
    def parse(
        self, 
        response: str, 
        doc_id: Optional[str] = None,
        source_text: Optional[str] = None,
        document_entity_ids: Optional[Set[str]] = None
    ) -> ParsedRelations:
        """
        Parse LLM response into structured relations.
        
        Args:
            response: LLM response text
            doc_id: Optional document ID
            source_text: Optional source text for entity resolution
            document_entity_ids: Optional set of entity IDs in the current document (limits entity resolution search space)
            
        Returns:
            ParsedRelations object
        """
        self.logger.info(f"[Parser] Parsing response for document: {doc_id}")
        parsed = ParsedRelations(doc_id=doc_id)
        
        # Try to extract JSON from response
        json_data = self._extract_json(response)
        
        if json_data:
            try:
                relations_data = json_data
                if isinstance(relations_data, dict) and "relations" in relations_data:
                    relations_data = relations_data["relations"]
                elif not isinstance(relations_data, list):
                    relations_data = [relations_data]
                
                for rel_data in relations_data:
                    if isinstance(rel_data, dict):
                        relation = ParsedRelation(
                            head_mention=rel_data.get("head_mention", "").strip(),
                            tail_mention=rel_data.get("tail_mention", "").strip(),
                            relation_type=rel_data.get("relation_type", "").strip(),
                            confidence=rel_data.get("confidence")
                        )
                        
                        if relation.head_mention and relation.tail_mention and relation.relation_type:
                            parsed.relations.append(relation)
                
                self.logger.info(f"[Parser] Extracted {len(parsed.relations)} relations from JSON")
                
            except Exception as e:
                error_msg = f"Error parsing JSON: {e}"
                parsed.parsing_errors.append(error_msg)
                self.logger.warning(f"[Parser] {error_msg}")
                # Log the full raw response at INFO level when JSON parsing fails
                self.logger.info(f"[Parser] Raw LLM response (JSON parsing failed, full output):\n{response}")
                self.logger.info(f"[Parser] Response length: {len(response)} characters")
        else:
            # Try text-based parsing as fallback
            error_msg = "No JSON found, attempting text parsing"
            parsed.parsing_errors.append(error_msg)
            self.logger.warning(f"[Parser] {error_msg}")
            # Log the full raw response at INFO level when JSON is not found
            self.logger.info(f"[Parser] Raw LLM response (no JSON found, full output):\n{response}")
            self.logger.info(f"[Parser] Response length: {len(response)} characters")
            text_relations = self._parse_text_format(response)
            parsed.relations.extend(text_relations)
            self.logger.info(f"[Parser] Extracted {len(text_relations)} relations from text format")
        
        # Log parsed relations
        self.logger.debug(f"[Parser] Parsed {len(parsed.relations)} relations:")
        for i, rel in enumerate(parsed.relations, 1):
            self.logger.debug(
                f"[Parser]   Relation {i}: {rel.head_mention} -> {rel.tail_mention} "
                f"({rel.relation_type})"
            )
        
        # Validate relations if validator is available and source text provided
        if self.validator and parsed.relations and source_text:
            self.logger.info(f"[Parser] Validating {len(parsed.relations)} relations against source text...")
            validated_relations, validation_errors = self.validator.validate_relations(
                parsed.relations,
                source_text,
                strict=False,
                filter_invalid=False  # Keep invalid ones but track errors
            )
            
            # Assign confidence scores
            validated_relations = self.validator.assign_confidence_scores(
                validated_relations,
                source_text
            )
            
            # Filter low-confidence relations (optional - can be configured)
            # For now, we keep all but log warnings
            low_confidence = [r for r in validated_relations if r.confidence and r.confidence < 0.5]
            if low_confidence:
                self.logger.warning(
                    f"[Parser] Found {len(low_confidence)} relations with low confidence (<0.5)"
                )
            
            parsed.relations = validated_relations
            parsed.parsing_errors.extend(validation_errors)
        
        # Resolve entity IDs if entity resolver is available
        if self.entity_resolver and parsed.relations:
            if document_entity_ids:
                self.logger.info(
                    f"[Parser] Resolving entity IDs for {len(parsed.relations)} relations "
                    f"(limited to {len(document_entity_ids)} entities in document)..."
                )
            else:
                self.logger.info(f"[Parser] Resolving entity IDs for {len(parsed.relations)} relations...")
            resolved_relations = self.entity_resolver.resolve_relations(
                parsed.relations,
                source_text=source_text,
                document_entity_ids=document_entity_ids
            )
            
            # Track resolution errors
            resolved_count = 0
            for relation in resolved_relations:
                if not relation.head_id:
                    error_msg = f"Could not resolve head entity: {relation.head_mention}"
                    parsed.entity_resolution_errors.append(error_msg)
                    self.logger.warning(f"[Parser] {error_msg}")
                else:
                    resolved_count += 1
                    
                if not relation.tail_id:
                    error_msg = f"Could not resolve tail entity: {relation.tail_mention}"
                    parsed.entity_resolution_errors.append(error_msg)
                    self.logger.warning(f"[Parser] {error_msg}")
                else:
                    resolved_count += 1
            
            self.logger.info(
                f"[Parser] Resolved {resolved_count}/{len(resolved_relations) * 2} entity IDs"
            )
            parsed.relations = resolved_relations
        
        # Log parsing errors if any
        if parsed.parsing_errors:
            self.logger.warning(f"[Parser] Parsing errors: {len(parsed.parsing_errors)}")
            for error in parsed.parsing_errors:
                self.logger.debug(f"[Parser]   Error: {error}")
        
        if parsed.entity_resolution_errors:
            self.logger.warning(f"[Parser] Entity resolution errors: {len(parsed.entity_resolution_errors)}")
            for error in parsed.entity_resolution_errors:
                self.logger.debug(f"[Parser]   Resolution Error: {error}")
        
        return parsed
    
    def _extract_json(self, text: str) -> Optional[dict]:
        """
        Extract JSON from text response.
        Handles markdown code fences, JSON arrays/objects, and nested structures.
        
        Args:
            text: Response text
            
        Returns:
            Parsed JSON dict or None
        """
        # First, try to extract JSON from markdown code fences (```json ... ``` or ``` ... ```)
        markdown_patterns = [
            r'```json\s*\n([\s\S]*?)\n```',  # ```json ... ```
            r'```\s*\n([\s\S]*?)\n```',      # ``` ... ```
            r'```json\s*([\s\S]*?)\n```',    # ```json...``` (no newline)
            r'```\s*([\s\S]*?)\n```',        # ```...``` (no newline)
        ]
        
        for pattern in markdown_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if parsed:  # Only return if we got something
                        return parsed
                except json.JSONDecodeError:
                    continue
        
        # Try to find JSON array or object by finding balanced brackets/braces
        # Look for arrays first (most common format)
        array_start = text.find('[')
        if array_start != -1:
            # Find matching closing bracket by counting brackets
            bracket_count = 0
            in_string = False
            escape_next = False
            
            for i in range(array_start, len(text)):
                char = text[i]
                
                if escape_next:
                    escape_next = False
                    continue
                
                if char == '\\':
                    escape_next = True
                    continue
                
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                
                if not in_string:
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            # Found matching bracket
                            json_str = text[array_start:i+1]
                            try:
                                parsed = json.loads(json_str)
                                if parsed:
                                    return parsed
                            except json.JSONDecodeError:
                                pass
                            break
        
        # Try object
        obj_start = text.find('{')
        if obj_start != -1:
            brace_count = 0
            in_string = False
            escape_next = False
            
            for i in range(obj_start, len(text)):
                char = text[i]
                
                if escape_next:
                    escape_next = False
                    continue
                
                if char == '\\':
                    escape_next = True
                    continue
                
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = text[obj_start:i+1]
                            try:
                                parsed = json.loads(json_str)
                                if parsed:
                                    return parsed
                            except json.JSONDecodeError:
                                pass
                            break
        
        # Try parsing the entire text as last resort
        try:
            parsed = json.loads(text.strip())
            if parsed:
                return parsed
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _parse_text_format(self, text: str) -> List[ParsedRelation]:
        """
        Parse relations from natural language text (fallback).
        
        Args:
            text: Response text
            
        Returns:
            List of ParsedRelation objects
        """
        relations = []
        
        # Look for patterns like "Entity1 -> Entity2: RelationType"
        pattern = r'([^->:]+)\s*->\s*([^->:]+)\s*:\s*([^\n]+)'
        matches = re.findall(pattern, text)
        
        for head, tail, rel_type in matches:
            relation = ParsedRelation(
                head_mention=head.strip(),
                tail_mention=tail.strip(),
                relation_type=rel_type.strip()
            )
            relations.append(relation)
        
        return relations
