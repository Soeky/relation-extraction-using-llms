"""Module for logging per-document evaluation results with relation classifications."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

from ..types import (
    EvaluationResult, 
    ParsedRelation, 
    Relation, 
    GoldRelations,
)
from ..data.entity_map import GlobalEntityMap


class DocumentLogger:
    """Logs per-document evaluation results with relation classifications."""
    
    def __init__(
        self,
        output_dir: Path,
        entity_map: Optional[GlobalEntityMap] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize document logger.
        
        Args:
            output_dir: Base output directory (will create technique subdirectories)
            entity_map: Optional global entity map for converting entity IDs to text
            logger: Optional logger instance
        """
        self.output_dir = output_dir
        self.entity_map = entity_map
        self.logger = logger or logging.getLogger(__name__)
    
    def log_document_results(
        self,
        technique_name: str,
        doc_id: str,
        eval_result: EvaluationResult,
        predicted_relations: List[ParsedRelation],
        gold_relations_obj: GoldRelations,
        stats: Dict[str, Any],
        prompt: str = "",
        raw_response: str = ""
    ) -> Path:
        """
        Log per-document results to a JSON file.
        
        Args:
            technique_name: Name of the prompting technique (e.g., "IO", "CoT")
            doc_id: Document ID
            eval_result: Evaluation result for this document
            predicted_relations: All predicted relations from LLM
            gold_relations_obj: Gold relations object (contains entities for text conversion)
            stats: Additional statistics dictionary
            
        Returns:
            Path to the saved JSON file
        """
        # Create technique-specific directory
        technique_dir = self.output_dir / technique_name
        technique_dir.mkdir(parents=True, exist_ok=True)
        
        # Classify predicted relations
        classified_relations = self._classify_relations(
            predicted_relations,
            eval_result
        )
        
        # Convert gold relations to text (include all gold relations)
        # Mark false negatives as omissions
        all_gold_relations = eval_result.false_negatives + eval_result.true_positives
        fn_set = {(rel.head_id, rel.tail_id, rel.type) for rel in eval_result.false_negatives}
        
        gold_relations_text = self._convert_gold_relations_to_text(
            all_gold_relations,
            gold_relations_obj,
            fn_set=fn_set
        )
        
        # Build structured document result with better organization
        document_result = self._build_structured_result(
            doc_id=doc_id,
            technique_name=technique_name,
            stats=stats,
            classified_relations=classified_relations,
            gold_relations_text=gold_relations_text,
            eval_result=eval_result,
            prompt=prompt,
            raw_response=raw_response
        )
        
        # Save to file
        output_file = technique_dir / f"{doc_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(document_result, f, indent=2, ensure_ascii=False)
        
        self.logger.debug(f"Saved document results to: {output_file}")
        return output_file
    
    def _classify_relations(
        self,
        predicted_relations: List[ParsedRelation],
        eval_result: EvaluationResult
    ) -> List[Dict[str, Any]]:
        """
        Classify each predicted relation as exact_match, hallucination, partial_match, or unresolved.
        
        Args:
            predicted_relations: All predicted relations
            eval_result: Evaluation result containing TP, FP, FN, partial matches
            
        Returns:
            List of classified relations with their status
        """
        # Build sets for quick lookup
        # True positives: exact matches (check both directions)
        tp_set: Set[Tuple[str, str, str]] = set()
        for rel in eval_result.true_positives:
            tp_set.add((rel.head_id, rel.tail_id, rel.type))
            tp_set.add((rel.tail_id, rel.head_id, rel.type))  # Bidirectional
        
        # Map partial matches: identify predicted relations that are partial matches
        # Use (head_id, tail_id) as key since entities match but type may differ
        partial_map: Dict[Tuple[str, str], Tuple[ParsedRelation, Relation]] = {}
        for pred_rel, gold_rel in eval_result.partial_matches:
            if pred_rel.head_id and pred_rel.tail_id:
                # Store in both directions (entities can match in either direction)
                partial_map[(pred_rel.head_id, pred_rel.tail_id)] = (pred_rel, gold_rel)
                partial_map[(pred_rel.tail_id, pred_rel.head_id)] = (pred_rel, gold_rel)
        
        # Map false positives by (head_id, tail_id, relation_type) for exact lookup
        fp_set: Set[Tuple[str, str, str]] = set()
        for fp in eval_result.false_positives:
            if fp.head_id and fp.tail_id:
                fp_set.add((fp.head_id, fp.tail_id, fp.relation_type))
                fp_set.add((fp.tail_id, fp.head_id, fp.relation_type))  # Bidirectional
        
        classified = []
        
        for pred_rel in predicted_relations:
            # Build relation dict
            rel_dict = {
                "head_mention": pred_rel.head_mention,
                "tail_mention": pred_rel.tail_mention,
                "relation_type": pred_rel.relation_type,
                "head_id": pred_rel.head_id,
                "tail_id": pred_rel.tail_id,
            }
            
            # Classify the relation
            if not pred_rel.head_id or not pred_rel.tail_id:
                rel_dict["status"] = "unresolved"
                rel_dict["status_description"] = "Entity resolution failed - could not match to entity IDs"
            else:
                # Check for exact match (true positive) - check both directions
                rel_tuple_forward = (pred_rel.head_id, pred_rel.tail_id, pred_rel.relation_type)
                rel_tuple_reverse = (pred_rel.tail_id, pred_rel.head_id, pred_rel.relation_type)
                
                if rel_tuple_forward in tp_set or rel_tuple_reverse in tp_set:
                    rel_dict["status"] = "exact_match"
                    rel_dict["status_description"] = "Exact match with gold relation"
                # Check for partial match (entities match, type differs)
                # Use (head_id, tail_id) key since entities match but type may differ
                elif (pred_rel.head_id, pred_rel.tail_id) in partial_map or (pred_rel.tail_id, pred_rel.head_id) in partial_map:
                    key = (pred_rel.head_id, pred_rel.tail_id) if (pred_rel.head_id, pred_rel.tail_id) in partial_map else (pred_rel.tail_id, pred_rel.head_id)
                    pred_partial, gold_partial = partial_map[key]
                    rel_dict["status"] = "partial_match"
                    rel_dict["status_description"] = f"Entities match but type differs: predicted '{pred_rel.relation_type}' vs gold '{gold_partial.type}'"
                    rel_dict["gold_relation_type"] = gold_partial.type
                # Check for hallucination (false positive)
                elif rel_tuple_forward in fp_set or rel_tuple_reverse in fp_set:
                    rel_dict["status"] = "hallucination"
                    rel_dict["status_description"] = "Relation not present in gold standard"
                else:
                    # This shouldn't happen, but handle it
                    rel_dict["status"] = "unknown"
                    rel_dict["status_description"] = "Could not classify relation"
            
            classified.append(rel_dict)
        
        return classified
    
    def _convert_gold_relations_to_text(
        self,
        gold_relations: List[Relation],
        gold_relations_obj: GoldRelations,
        fn_set: Optional[Set[Tuple[str, str, str]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Convert gold relations (with entity IDs) to text format.
        
        Args:
            gold_relations: List of gold Relation objects
            gold_relations_obj: GoldRelations object containing entities with mentions
            fn_set: Optional set of (head_id, tail_id, type) tuples for false negatives (omissions)
            
        Returns:
            List of gold relations with text mentions and status
        """
        # Build entity ID to text mapping from gold relations
        entity_id_to_text: Dict[str, str] = {}
        for entity in gold_relations_obj.entities:
            if entity.mentions:
                # Use first mention as the text representation
                entity_id_to_text[entity.id] = entity.mentions[0].text
            elif self.entity_map:
                # Fallback to entity map
                global_entity = self.entity_map.get_entity(entity.id)
                if global_entity and global_entity.canonical_name:
                    entity_id_to_text[entity.id] = global_entity.canonical_name
                elif global_entity and global_entity.common_mentions:
                    entity_id_to_text[entity.id] = global_entity.common_mentions[0]
                else:
                    entity_id_to_text[entity.id] = entity.id  # Fallback to ID
            else:
                entity_id_to_text[entity.id] = entity.id  # Fallback to ID
        
        # Convert relations
        gold_relations_text = []
        for rel in gold_relations:
            head_text = entity_id_to_text.get(rel.head_id, rel.head_id)
            tail_text = entity_id_to_text.get(rel.tail_id, rel.tail_id)
            
            # Check if this is a false negative (omission)
            is_omission = False
            if fn_set is not None:
                rel_tuple = (rel.head_id, rel.tail_id, rel.type)
                is_omission = rel_tuple in fn_set
            
            gold_rel_dict = {
                "head_id": rel.head_id,
                "head_mention": head_text,
                "tail_id": rel.tail_id,
                "tail_mention": tail_text,
                "relation_type": rel.type,
                "relation_id": rel.id,
                "novel": rel.novel,
                "status": "omission" if is_omission else "found",
                "status_description": "Not found in LLM output (false negative)" if is_omission else "Found in LLM output (true positive)"
            }
            
            gold_relations_text.append(gold_rel_dict)
        
        return gold_relations_text
    
    def _build_structured_result(
        self,
        doc_id: str,
        technique_name: str,
        stats: Dict[str, Any],
        classified_relations: List[Dict[str, Any]],
        gold_relations_text: List[Dict[str, Any]],
        eval_result: EvaluationResult,
        prompt: str = "",
        raw_response: str = ""
    ) -> Dict[str, Any]:
        """
        Build a structured result with better organization for comparison and analysis.
        
        Args:
            doc_id: Document ID
            technique_name: Technique name
            stats: Statistics dictionary
            classified_relations: Classified predicted relations
            gold_relations_text: Gold relations with text
            eval_result: Evaluation result
            
        Returns:
            Structured document result dictionary
        """
        # Group relations by status
        relations_by_status = {
            "exact_match": [r for r in classified_relations if r["status"] == "exact_match"],
            "partial_match": [r for r in classified_relations if r["status"] == "partial_match"],
            "hallucination": [r for r in classified_relations if r["status"] == "hallucination"],
            "unresolved": [r for r in classified_relations if r["status"] == "unresolved"],
        }
        
        gold_by_status = {
            "found": [r for r in gold_relations_text if r["status"] == "found"],
            "omission": [r for r in gold_relations_text if r["status"] == "omission"],
        }
        
        # Build comparison pairs
        comparisons = self._build_comparisons(classified_relations, gold_relations_text, eval_result)
        
        # Analyze issues
        analysis = self._analyze_issues(classified_relations, gold_relations_text, stats)
        
        # Build structured result
        result = {
            "metadata": {
                "doc_id": doc_id,
                "technique": technique_name,
                "summary": {
                    "total_predicted": len(classified_relations),
                    "total_gold": len(gold_relations_text),
                    "exact_matches": len(relations_by_status["exact_match"]),
                    "partial_matches": len(relations_by_status["partial_match"]),
                    "hallucinations": len(relations_by_status["hallucination"]),
                    "unresolved": len(relations_by_status["unresolved"]),
                    "omissions": len(gold_by_status["omission"]),
                }
            },
            "prompt": prompt,  # Log the actual prompt used
            "raw_llm_response": raw_response,  # Log the complete raw LLM response (including all reasoning, not just JSON)
            "stats": stats,
            "analysis": analysis,
            "grouped_relations": {
                "predicted": {
                    "exact_matches": relations_by_status["exact_match"],
                    "partial_matches": relations_by_status["partial_match"],
                    "hallucinations": relations_by_status["hallucination"],
                    "unresolved": relations_by_status["unresolved"],
                },
                "gold": {
                    "found": gold_by_status["found"],
                    "omissions": gold_by_status["omission"],
                }
            },
            "comparisons": comparisons,
            # Keep full lists for backward compatibility
            "llm_generated_relations": classified_relations,
            "golden_relations": gold_relations_text,
        }
        
        return result
    
    def _build_comparisons(
        self,
        classified_relations: List[Dict[str, Any]],
        gold_relations_text: List[Dict[str, Any]],
        eval_result: EvaluationResult
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build side-by-side comparisons for easier analysis.
        
        Returns:
            Dictionary with comparison groups
        """
        comparisons = {
            "exact_matches": [],
            "partial_matches": [],
            "missed_relations": [],
            "hallucinated_relations": [],
        }
        
        # Map gold relations by (head_id, tail_id, type) for quick lookup
        gold_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for gold_rel in gold_relations_text:
            key = (gold_rel["head_id"], gold_rel["tail_id"], gold_rel["relation_type"])
            gold_map[key] = gold_rel
            # Also add reverse direction
            gold_map[(gold_rel["tail_id"], gold_rel["head_id"], gold_rel["relation_type"])] = gold_rel
        
        # Build exact match comparisons
        for pred_rel in classified_relations:
            if pred_rel["status"] == "exact_match" and pred_rel["head_id"] and pred_rel["tail_id"]:
                key = (pred_rel["head_id"], pred_rel["tail_id"], pred_rel["relation_type"])
                gold_rel = gold_map.get(key)
                if gold_rel:
                    comparisons["exact_matches"].append({
                        "predicted": {
                            "head": pred_rel["head_mention"],
                            "tail": pred_rel["tail_mention"],
                            "type": pred_rel["relation_type"],
                        },
                        "gold": {
                            "head": gold_rel["head_mention"],
                            "tail": gold_rel["tail_mention"],
                            "type": gold_rel["relation_type"],
                            "novel": gold_rel.get("novel", "No"),
                        }
                    })
        
        # Build partial match comparisons using eval_result
        gold_id_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for gold_rel in gold_relations_text:
            key = (gold_rel["head_id"], gold_rel["tail_id"], gold_rel["relation_type"])
            gold_id_map[key] = gold_rel
        
        for pred_rel, gold_rel_obj in eval_result.partial_matches:
            if pred_rel.head_id and pred_rel.tail_id:
                # Find the gold relation in our text format
                gold_key = (gold_rel_obj.head_id, gold_rel_obj.tail_id, gold_rel_obj.type)
                matching_gold = gold_id_map.get(gold_key)
                
                if matching_gold:
                    comparisons["partial_matches"].append({
                        "predicted": {
                            "head": pred_rel.head_mention,
                            "tail": pred_rel.tail_mention,
                            "type": pred_rel.relation_type,
                        },
                        "gold": {
                            "head": matching_gold["head_mention"],
                            "tail": matching_gold["tail_mention"],
                            "type": matching_gold["relation_type"],
                            "novel": matching_gold.get("novel", "No"),
                        },
                        "issue": f"Type mismatch: predicted '{pred_rel.relation_type}' vs gold '{gold_rel_obj.type}'"
                    })
        
        # Build missed relations (omissions)
        for gold_rel in gold_relations_text:
            if gold_rel["status"] == "omission":
                comparisons["missed_relations"].append({
                    "gold": {
                        "head": gold_rel["head_mention"],
                        "tail": gold_rel["tail_mention"],
                        "type": gold_rel["relation_type"],
                        "novel": gold_rel.get("novel", "No"),
                        "relation_id": gold_rel.get("relation_id"),
                    },
                    "issue": "Not found in LLM output"
                })
        
        # Build hallucinated relations
        for pred_rel in classified_relations:
            if pred_rel["status"] == "hallucination":
                comparisons["hallucinated_relations"].append({
                    "predicted": {
                        "head": pred_rel["head_mention"],
                        "tail": pred_rel["tail_mention"],
                        "type": pred_rel["relation_type"],
                        "head_id": pred_rel.get("head_id"),
                        "tail_id": pred_rel.get("tail_id"),
                    },
                    "issue": "Not present in gold standard"
                })
        
        return comparisons
    
    def _analyze_issues(
        self,
        classified_relations: List[Dict[str, Any]],
        gold_relations_text: List[Dict[str, Any]],
        stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze common issues and patterns.
        
        Returns:
            Analysis dictionary with insights
        """
        analysis = {
            "entity_resolution_issues": {
                "unresolved_count": len([r for r in classified_relations if r["status"] == "unresolved"]),
                "unresolved_relations": [
                    {
                        "head_mention": r["head_mention"],
                        "tail_mention": r["tail_mention"],
                        "type": r["relation_type"],
                        "missing_head_id": r.get("head_id") is None,
                        "missing_tail_id": r.get("tail_id") is None,
                    }
                    for r in classified_relations if r["status"] == "unresolved"
                ]
            },
            "relation_type_analysis": {
                "predicted_types": {},
                "gold_types": {},
                "type_mismatches": [],
            },
            "novel_relations_analysis": {
                "missed_novel": 0,
                "missed_known": 0,
                "hallucinated_novel_like": 0,
            },
            "key_insights": []
        }
        
        # Count relation types
        for rel in classified_relations:
            rel_type = rel["relation_type"]
            analysis["relation_type_analysis"]["predicted_types"][rel_type] = \
                analysis["relation_type_analysis"]["predicted_types"].get(rel_type, 0) + 1
        
        for rel in gold_relations_text:
            rel_type = rel["relation_type"]
            analysis["relation_type_analysis"]["gold_types"][rel_type] = \
                analysis["relation_type_analysis"]["gold_types"].get(rel_type, 0) + 1
        
        # Analyze type mismatches
        for rel in classified_relations:
            if rel["status"] == "partial_match" and "gold_relation_type" in rel:
                analysis["relation_type_analysis"]["type_mismatches"].append({
                    "predicted_type": rel["relation_type"],
                    "gold_type": rel["gold_relation_type"],
                    "head": rel["head_mention"],
                    "tail": rel["tail_mention"],
                })
        
        # Analyze novel relations
        for gold_rel in gold_relations_text:
            if gold_rel["status"] == "omission":
                if gold_rel.get("novel") == "Novel":
                    analysis["novel_relations_analysis"]["missed_novel"] += 1
                else:
                    analysis["novel_relations_analysis"]["missed_known"] += 1
        
        # Generate key insights
        insights = []
        
        if stats.get("omission_rate", 0) > 0.7:
            insights.append(f"High omission rate ({stats['omission_rate']:.1%}): Many gold relations were missed")
        
        if stats.get("hallucination_rate", 0) > 0.5:
            insights.append(f"High hallucination rate ({stats['hallucination_rate']:.1%}): Many incorrect relations predicted")
        
        if analysis["entity_resolution_issues"]["unresolved_count"] > 0:
            insights.append(f"{analysis['entity_resolution_issues']['unresolved_count']} relations had entity resolution failures")
        
        if len(analysis["relation_type_analysis"]["type_mismatches"]) > 0:
            insights.append(f"{len(analysis['relation_type_analysis']['type_mismatches'])} relations had correct entities but wrong type")
        
        if analysis["novel_relations_analysis"]["missed_novel"] > 0:
            insights.append(f"Missed {analysis['novel_relations_analysis']['missed_novel']} novel relations")
        
        analysis["key_insights"] = insights
        
        return analysis

