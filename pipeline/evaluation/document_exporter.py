"""Document exporter for detailed per-document JSON output."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..types import (
    EvaluationResult,
    GoldRelations,
    ParsedRelations,
    Relation,
    ParsedRelation
)


class DocumentExporter:
    """Exports detailed per-document evaluation results to JSON."""
    
    @staticmethod
    def export_document_details(
        doc_id: str,
        strategy: str,
        predicted_relations: ParsedRelations,
        gold_relations_obj: GoldRelations,
        evaluation_result: EvaluationResult,
        output_path: Path
    ):
        """
        Export detailed per-document evaluation results to JSON.
        
        Args:
            doc_id: Document ID
            strategy: Matching strategy name
            predicted_relations: Predicted relations from LLM
            gold_relations_obj: Gold relations object
            evaluation_result: Evaluation result with metrics
            output_path: Path to save JSON file
        """
        # Build detailed matches list
        matches = []
        
        # True positives: matched predicted relations with their gold counterparts
        matched_predicted_ids = set()
        for tp in evaluation_result.true_positives:
            # Find corresponding predicted relation
            for pred_rel in predicted_relations.relations:
                pred_key = (pred_rel.head_mention, pred_rel.tail_mention, pred_rel.relation_type)
                if pred_key not in matched_predicted_ids:
                    # Check if this predicted relation matches the TP
                    # (This is a simplified check - in practice, we'd use the semantic_matches data)
                    gold_key = (tp.head_id, tp.tail_id, tp.type)
                    # Try to find in semantic matches first
                    found_match = False
                    similarity_score = 1.0
                    entity_similarities = {}
                    
                    for pred, gold, score in evaluation_result.semantic_matches:
                        if gold.id == tp.id:
                            similarity_score = score
                            found_match = True
                            # Try to extract entity similarities if available
                            entity_similarities = {"head": score, "tail": score}
                            break
                    
                    if found_match or (pred_rel.head_id == tp.head_id and pred_rel.tail_id == tp.tail_id):
                        matches.append({
                            "predicted": {
                                "head_mention": pred_rel.head_mention,
                                "tail_mention": pred_rel.tail_mention,
                                "relation_type": pred_rel.relation_type,
                                "head_id": pred_rel.head_id,
                                "tail_id": pred_rel.tail_id
                            },
                            "gold": {
                                "head_id": tp.head_id,
                                "tail_id": tp.tail_id,
                                "relation_type": tp.type,
                                "relation_id": tp.id
                            },
                            "match_type": "true_positive",
                            "similarity_score": similarity_score,
                            "entity_similarities": entity_similarities
                        })
                        matched_predicted_ids.add(pred_key)
                        break
        
        # False positives: predicted relations with no match
        for fp in evaluation_result.false_positives:
            fp_key = (fp.head_mention, fp.tail_mention, fp.relation_type)
            if fp_key not in matched_predicted_ids:
                matches.append({
                    "predicted": {
                        "head_mention": fp.head_mention,
                        "tail_mention": fp.tail_mention,
                        "relation_type": fp.relation_type,
                        "head_id": fp.head_id,
                        "tail_id": fp.tail_id
                    },
                    "gold": None,
                    "match_type": "false_positive",
                    "similarity_score": 0.0,
                    "entity_similarities": {}
                })
        
        # False negatives: gold relations with no match
        matched_gold_ids = {tp.id for tp in evaluation_result.true_positives}
        matched_gold_ids.update({gold.id for _, gold, _ in evaluation_result.semantic_matches})
        
        for fn in evaluation_result.false_negatives:
            if fn.id not in matched_gold_ids:
                matches.append({
                    "predicted": None,
                    "gold": {
                        "head_id": fn.head_id,
                        "tail_id": fn.tail_id,
                        "relation_type": fn.type,
                        "relation_id": fn.id
                    },
                    "match_type": "false_negative",
                    "similarity_score": 0.0,
                    "entity_similarities": {}
                })
        
        # Semantic matches (partial matches with scores)
        for pred, gold, score in evaluation_result.semantic_matches:
            matches.append({
                "predicted": {
                    "head_mention": pred.head_mention,
                    "tail_mention": pred.tail_mention,
                    "relation_type": pred.relation_type,
                    "head_id": pred.head_id,
                    "tail_id": pred.tail_id
                },
                "gold": {
                    "head_id": gold.head_id,
                    "tail_id": gold.tail_id,
                    "relation_type": gold.type,
                    "relation_id": gold.id
                },
                "match_type": "semantic_match",
                "similarity_score": score,
                "entity_similarities": {"head": score, "tail": score}  # Simplified
            })
        
        # Build gold relations list with entity mentions
        gold_relations_list = []
        for gold_rel in gold_relations_obj.relations:
            # Get entity mentions
            head_entity = None
            tail_entity = None
            for entity in gold_relations_obj.entities:
                if entity.id == gold_rel.head_id:
                    head_entity = entity
                if entity.id == gold_rel.tail_id:
                    tail_entity = entity
            
            head_mentions = [m.text for m in head_entity.mentions] if head_entity and head_entity.mentions else []
            tail_mentions = [m.text for m in tail_entity.mentions] if tail_entity and tail_entity.mentions else []
            
            gold_relations_list.append({
                "relation_id": gold_rel.id,
                "head_id": gold_rel.head_id,
                "tail_id": gold_rel.tail_id,
                "relation_type": gold_rel.type,
                "head_mentions": head_mentions if head_mentions else [gold_rel.head_id],
                "tail_mentions": tail_mentions if tail_mentions else [gold_rel.tail_id],
                "novel": gold_rel.novel
            })
        
        # Build predicted relations list
        predicted_relations_list = []
        for pred_rel in predicted_relations.relations:
            predicted_relations_list.append({
                "head_mention": pred_rel.head_mention,
                "tail_mention": pred_rel.tail_mention,
                "relation_type": pred_rel.relation_type,
                "head_id": pred_rel.head_id,
                "tail_id": pred_rel.tail_id,
                "confidence": pred_rel.confidence
            })
        
        # Build summary
        summary = {
            "tp_count": len(evaluation_result.true_positives),
            "fp_count": len(evaluation_result.false_positives),
            "fn_count": len(evaluation_result.false_negatives),
            "semantic_match_count": len(evaluation_result.semantic_matches),
            "precision": evaluation_result.precision,
            "recall": evaluation_result.recall,
            "f1_score": evaluation_result.f1_score,
            "fuzzy_precision": evaluation_result.fuzzy_precision,
            "fuzzy_recall": evaluation_result.fuzzy_recall,
            "fuzzy_f1": evaluation_result.fuzzy_f1,
            "exact_match_rate": evaluation_result.exact_match_rate,
            "omission_rate": evaluation_result.omission_rate,
            "hallucination_rate": evaluation_result.hallucination_rate,
            "graph_edit_distance": evaluation_result.graph_edit_distance,
            "bertscore": evaluation_result.bertscore
        }
        
        # Build final document data
        document_data = {
            "doc_id": doc_id,
            "strategy": strategy,
            "gold_relations": gold_relations_list,
            "predicted_relations": predicted_relations_list,
            "matches": matches,
            "summary": summary
        }
        
        # Save to JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(document_data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def export_all_documents(
        evaluation_results: List[EvaluationResult],
        predicted_relations_list: List[ParsedRelations],
        gold_relations_list: List[GoldRelations],
        output_dir: Path,
        strategy: str
    ):
        """
        Export detailed results for all documents.
        
        Args:
            evaluation_results: List of evaluation results (one per document)
            predicted_relations_list: List of predicted relations (one per document)
            gold_relations_list: List of gold relations (one per document)
            output_dir: Output directory for JSON files
            strategy: Matching strategy name
        """
        if len(evaluation_results) != len(predicted_relations_list) or \
           len(evaluation_results) != len(gold_relations_list):
            raise ValueError("Lists must have the same length")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for eval_result, pred_rels, gold_rels in zip(
            evaluation_results, predicted_relations_list, gold_relations_list
        ):
            output_path = output_dir / f"{eval_result.doc_id}_{strategy}.json"
            DocumentExporter.export_document_details(
                eval_result.doc_id,
                strategy,
                pred_rels,
                gold_rels,
                eval_result,
                output_path
            )

