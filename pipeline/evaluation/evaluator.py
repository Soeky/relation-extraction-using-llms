"""Main evaluator class."""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..types import ParsedRelations, GoldRelations, EvaluationResult
from .metrics import MetricsCalculator
from .matchers.registry import MatcherRegistry, create_matcher
from config import Config


class Evaluator:
    """Evaluates predicted relations against gold standard."""
    
    def __init__(
        self, 
        strategy: str = "text",
        entity_map=None, 
        match_type: bool = True,
        matcher_config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            strategy: Matching strategy name (e.g., "text", "jaccard", "bertscore")
                     Must be a registered matcher name
            entity_map: Optional global entity map
            match_type: Whether to require relation type to match
            matcher_config: Optional configuration dictionary for the matcher
            logger: Optional logger instance
        """
        self.strategy = strategy
        self.entity_map = entity_map
        self.match_type = match_type
        self.metrics_calculator = MetricsCalculator()
        self.logger = logger or logging.getLogger(__name__)
        
        # Create matcher using registry
        try:
            config = matcher_config or {}
            config['match_type'] = match_type
            self.matcher = create_matcher(
                name=strategy,
                config=config,
                entity_map=entity_map,
                logger=logger
            )
            self.logger.info(
                f"[Evaluator] Using strategy '{strategy}' with matcher: {self.matcher.get_name()}"
            )
        except Exception as e:
            self.logger.error(
                f"[Evaluator] Failed to create matcher '{strategy}': {e}. "
                f"Available strategies: {MatcherRegistry.get_available_matchers()}"
            )
            raise
    
    def evaluate(
        self,
        predictions: List[ParsedRelations],
        gold_relations_list: List[GoldRelations]
    ) -> List[EvaluationResult]:
        """
        Evaluate predictions against gold standard.
        
        Args:
            predictions: List of ParsedRelations (one per document)
            gold_relations_list: List of GoldRelations (one per document)
            
        Returns:
            List of EvaluationResult objects
        """
        if len(predictions) != len(gold_relations_list):
            raise ValueError(
                f"Mismatch: {len(predictions)} predictions vs "
                f"{len(gold_relations_list)} gold relations"
            )
        
        results = []
        
        for pred, gold in zip(predictions, gold_relations_list):
            doc_id = gold.doc_id
            gold_file_name = Path(gold.file_path).name if gold.file_path else "unknown"
            self.logger.info(
                f"[Evaluator] Evaluating document: {doc_id} "
                f"(strategy: {self.strategy}, gold relations file: {gold_file_name})"
            )
            
            if pred.doc_id != gold.doc_id:
                self.logger.warning(
                    f"[Evaluator] Doc ID mismatch: pred={pred.doc_id}, gold={gold.doc_id} "
                    f"(gold file: {gold_file_name})"
                )
            
            # Convert ParsedRelations to list of ParsedRelation
            predicted_relations = pred.relations
            gold_relations = gold.relations
            
            self.logger.info(
                f"[Evaluator] Document {doc_id}: "
                f"{len(predicted_relations)} predicted relations, "
                f"{len(gold_relations)} gold relations"
            )
            
            # Debug: Show LLM relations and gold relations
            self.logger.debug(f"\n[Evaluator] Document {doc_id} - LLM Predicted Relations:")
            if predicted_relations:
                for i, rel in enumerate(predicted_relations, 1):
                    head_display = rel.head_mention
                    tail_display = rel.tail_mention
                    if rel.head_id:
                        head_display += f" (ID: {rel.head_id})"
                    else:
                        head_display += " [RESOLUTION ERROR]"
                    if rel.tail_id:
                        tail_display += f" (ID: {rel.tail_id})"
                    else:
                        tail_display += " [RESOLUTION ERROR]"
                    
                    self.logger.debug(
                        f"  {i}. {head_display} -> {tail_display} "
                        f"({rel.relation_type})"
                    )
            else:
                self.logger.debug("  (no relations predicted)")
            
            self.logger.debug(f"\n[Evaluator] Document {doc_id} - Gold Relations:")
            if gold_relations:
                for i, rel in enumerate(gold_relations, 1):
                    # Try to resolve entity IDs to mentions using entity map
                    head_display = rel.head_id
                    tail_display = rel.tail_id
                    
                    if self.entity_map:
                        head_entity = self.entity_map.get_entity(rel.head_id)
                        tail_entity = self.entity_map.get_entity(rel.tail_id)
                        
                        if head_entity and head_entity.canonical_name:
                            head_display = f"{head_entity.canonical_name} (ID: {rel.head_id})"
                        elif head_entity and head_entity.common_mentions:
                            head_display = f"{head_entity.common_mentions[0]} (ID: {rel.head_id})"
                        else:
                            head_display = f"{rel.head_id} [ENTITY NOT FOUND IN MAP]"
                        
                        if tail_entity and tail_entity.canonical_name:
                            tail_display = f"{tail_entity.canonical_name} (ID: {rel.tail_id})"
                        elif tail_entity and tail_entity.common_mentions:
                            tail_display = f"{tail_entity.common_mentions[0]} (ID: {rel.tail_id})"
                        else:
                            tail_display = f"{rel.tail_id} [ENTITY NOT FOUND IN MAP]"
                    
                    self.logger.debug(
                        f"  {i}. {head_display} -> {tail_display} "
                        f"({rel.type}) [relation_id: {rel.id}]"
                    )
            else:
                self.logger.debug("  (no gold relations)")
            
            # Match relations using the matcher
            self.logger.info(
                f"[Evaluator] Using strategy '{self.strategy}' for document {doc_id} "
                f"(match_type={self.match_type})"
            )
            
            true_positives, false_positives, false_negatives, semantic_matches = \
                self.matcher.match(
                    predicted_relations,
                    gold,
                    match_type=self.match_type
                )
            
            # Extract partial matches from semantic_matches (those with lower scores)
            # Semantic matches are (predicted, gold, similarity_score) tuples
            partial_matches = []
            high_confidence_semantic_matches = []
            
            for pred, gold_rel, score in semantic_matches:
                # If score is high enough, it might be a partial match (entity match, type mismatch)
                # Otherwise it's a semantic match
                if score >= 0.5 and score < 0.95:  # Threshold for partial vs semantic
                    partial_matches.append((pred, gold_rel))
                else:
                    high_confidence_semantic_matches.append((pred, gold_rel, score))
            
            tp_count = len(true_positives)
            fp_count = len(false_positives)
            fn_count = len(false_negatives)
            partial_count = len(partial_matches)
            semantic_count = len(semantic_matches)
            
            self.logger.info(
                f"[Evaluator] Document {doc_id} matching results: "
                f"TP={tp_count}, FP={fp_count}, FN={fn_count}, "
                f"Partial Matches={partial_count}, Semantic Matches={semantic_count}"
            )
            
            # Log partial matches (entities match but type differs)
            if partial_matches:
                self.logger.debug(f"\n[Evaluator] Document {doc_id} - Partial Matches:")
                for i, (pred, gold) in enumerate(partial_matches[:5], 1):  # Show first 5
                    head_name = pred.head_mention
                    tail_name = pred.tail_mention
                    if self.entity_map:
                        head_entity = self.entity_map.get_entity(pred.head_id)
                        tail_entity = self.entity_map.get_entity(pred.tail_id)
                        if head_entity and head_entity.canonical_name:
                            head_name = head_entity.canonical_name
                        if tail_entity and tail_entity.canonical_name:
                            tail_name = tail_entity.canonical_name
                    
                    self.logger.debug(
                        f"  {i}. Predicted: {head_name} -> {tail_name} ({pred.relation_type}) | "
                        f"Gold: {gold.type} [entities match, type differs]"
                    )
            
            # Log some examples
            if tp_count > 0:
                self.logger.debug(f"[Evaluator] Document {doc_id} True Positives (first 3):")
                for i, tp in enumerate(true_positives[:3], 1):
                    self.logger.debug(
                        f"[Evaluator]   TP {i}: {tp.head_id} -> {tp.tail_id} ({tp.type})"
                    )
            
            if fp_count > 0:
                self.logger.debug(f"[Evaluator] Document {doc_id} False Positives (first 3):")
                for i, fp in enumerate(false_positives[:3], 1):
                    self.logger.debug(
                        f"[Evaluator]   FP {i}: {fp.head_mention} -> {fp.tail_mention} "
                        f"({fp.relation_type})"
                    )
            
            if fn_count > 0:
                self.logger.debug(f"[Evaluator] Document {doc_id} False Negatives (first 3):")
                for i, fn in enumerate(false_negatives[:3], 1):
                    self.logger.debug(
                        f"[Evaluator]   FN {i}: {fn.head_id} -> {fn.tail_id} ({fn.type})"
                    )
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_metrics(
                true_positives=true_positives,
                false_positives=false_positives,
                false_negatives=false_negatives,
                gold_relations=gold_relations,
                predicted_relations=predicted_relations
            )
            
            # Calculate fuzzy metrics (treating partial matches and semantic matches as correct for entities)
            fuzzy_tp = tp_count + partial_count + len(high_confidence_semantic_matches)
            fuzzy_fp = fp_count - partial_count - len(high_confidence_semantic_matches)
            fuzzy_fn = fn_count
            
            fuzzy_precision = fuzzy_tp / (fuzzy_tp + fuzzy_fp) if (fuzzy_tp + fuzzy_fp) > 0 else 0.0
            fuzzy_recall = fuzzy_tp / (fuzzy_tp + fuzzy_fn) if (fuzzy_tp + fuzzy_fn) > 0 else 0.0
            fuzzy_f1 = (
                2 * (fuzzy_precision * fuzzy_recall) / (fuzzy_precision + fuzzy_recall)
                if (fuzzy_precision + fuzzy_recall) > 0 else 0.0
            )
            
            # Compute average similarity score for semantic matches
            if semantic_matches:
                bertscore_value = sum(score for _, _, score in semantic_matches) / len(semantic_matches)
            else:
                bertscore_value = 0.0
            
            # Build detailed matches for export
            detailed_matches = []
            matched_pred_ids = set()
            matched_gold_ids = {tp.id for tp in true_positives}
            
            # Add true positives
            for tp in true_positives:
                matched_gold_ids.add(tp.id)
                # Find corresponding predicted relation
                for pred_rel in predicted_relations:
                    pred_key = (pred_rel.head_mention, pred_rel.tail_mention, pred_rel.relation_type)
                    if pred_key not in matched_pred_ids:
                        # Check if matches
                        if (pred_rel.head_id == tp.head_id and pred_rel.tail_id == tp.tail_id) or \
                           (pred_rel.head_id == tp.tail_id and pred_rel.tail_id == tp.head_id):
                            detailed_matches.append({
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
                                "similarity_score": 1.0,
                                "entity_similarities": {"head": 1.0, "tail": 1.0}
                            })
                            matched_pred_ids.add(pred_key)
                            break
            
            # Add semantic matches
            for pred, gold, score in semantic_matches:
                matched_gold_ids.add(gold.id)
                pred_key = (pred.head_mention, pred.tail_mention, pred.relation_type)
                matched_pred_ids.add(pred_key)
                detailed_matches.append({
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
                    "entity_similarities": {"head": score, "tail": score}
                })
            
            # Add false positives
            for fp in false_positives:
                pred_key = (fp.head_mention, fp.tail_mention, fp.relation_type)
                if pred_key not in matched_pred_ids:
                    detailed_matches.append({
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
            
            # Add false negatives
            for fn in false_negatives:
                if fn.id not in matched_gold_ids:
                    detailed_matches.append({
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
            
            # Create evaluation result
            result = EvaluationResult(
                doc_id=doc_id,
                strategy=self.strategy,
                true_positives=true_positives,
                false_positives=false_positives,
                false_negatives=false_negatives,
                partial_matches=partial_matches,
                semantic_matches=semantic_matches,
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1_score'],
                fuzzy_precision=fuzzy_precision,
                fuzzy_recall=fuzzy_recall,
                fuzzy_f1=fuzzy_f1,
                exact_match_rate=metrics['exact_match_rate'],
                omission_rate=metrics['omission_rate'],
                hallucination_rate=metrics['hallucination_rate'],
                redundancy_rate=metrics['redundancy_rate'],
                graph_edit_distance=metrics['graph_edit_distance'],
                bertscore=bertscore_value,
                per_type_metrics=metrics['per_type_metrics'],
                detailed_matches=detailed_matches
            )
            
            self.logger.info(
                f"[Evaluator] Document {doc_id} metrics: "
                f"Precision={result.precision:.3f}, "
                f"Recall={result.recall:.3f}, "
                f"F1={result.f1_score:.3f}"
            )
            
            results.append(result)
        
        return results
