"""Result aggregator for aggregating results across documents."""

import logging
import statistics
from typing import List, Optional

from ..types import EvaluationResult, AggregateResults


class ResultAggregator:
    """Aggregates evaluation results across documents."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize aggregator.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def aggregate(
        self,
        eval_results: List[EvaluationResult],
        technique_name: str,
        exclude_failed: bool = True
    ) -> AggregateResults:
        """
        Aggregate evaluation results across documents.
        
        Args:
            eval_results: List of per-document evaluation results
            technique_name: Name of the prompting technique
            exclude_failed: If True, exclude documents where LLM returned 0 predictions
                           but gold had relations (likely API failure). Default: True
            
        Returns:
            AggregateResults object with aggregated metrics
        """
        if not eval_results:
            return AggregateResults(
                technique_name=technique_name,
                per_document_results=eval_results
            )
        
        # Filter out failed documents if requested
        original_count = len(eval_results)
        if exclude_failed:
            # A document is considered "failed" if:
            # - It has 0 predicted relations (TP=0, FP=0)
            # - But it has gold relations (FN > 0)
            # This indicates the LLM failed to return anything, not that there were no relations
            valid_results = []
            failed_docs = []
            
            for result in eval_results:
                num_predicted = len(result.true_positives) + len(result.false_positives)
                num_gold = len(result.true_positives) + len(result.false_negatives)
                
                # If no predictions but there were gold relations, likely a failure
                if num_predicted == 0 and num_gold > 0:
                    failed_docs.append(result.doc_id)
                else:
                    valid_results.append(result)
            
            if failed_docs:
                self.logger.warning(
                    f"[Aggregator] Excluding {len(failed_docs)} failed documents from {technique_name} stats: "
                    f"{failed_docs[:5]}{'...' if len(failed_docs) > 5 else ''}"
                )
            
            eval_results = valid_results
        
        if not eval_results:
            self.logger.warning(
                f"[Aggregator] No valid results for {technique_name} after filtering "
                f"(all {original_count} documents failed)"
            )
            return AggregateResults(
                technique_name=technique_name,
                per_document_results=[]
            )
        
        n = len(eval_results)
        
        if n < original_count:
            self.logger.info(
                f"[Aggregator] {technique_name}: Using {n}/{original_count} documents "
                f"({original_count - n} excluded due to API failures)"
            )
        
        # Macro averages (average of per-document metrics)
        macro_precision = sum(r.precision for r in eval_results) / n
        macro_recall = sum(r.recall for r in eval_results) / n
        macro_f1 = sum(r.f1_score for r in eval_results) / n
        
        # Micro averages (calculated from aggregated TP/FP/FN)
        total_tp = sum(len(r.true_positives) for r in eval_results)
        total_fp = sum(len(r.false_positives) for r in eval_results)
        total_fn = sum(len(r.false_negatives) for r in eval_results)
        total_gold = total_tp + total_fn  # All gold relations across all documents
        total_predicted = total_tp + total_fp  # All predicted relations across all documents
        
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (
            2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0 else 0.0
        )
        
        # Overall aggregated rates (calculated from totals, not averaged)
        overall_exact_match_rate = total_tp / total_gold if total_gold > 0 else 0.0
        overall_omission_rate = total_fn / total_gold if total_gold > 0 else 0.0
        overall_hallucination_rate = total_fp / total_predicted if total_predicted > 0 else 0.0
        
        # Average rates
        avg_exact_match_rate = sum(r.exact_match_rate for r in eval_results) / n
        avg_omission_rate = sum(r.omission_rate for r in eval_results) / n
        avg_hallucination_rate = sum(r.hallucination_rate for r in eval_results) / n
        avg_redundancy_rate = sum(r.redundancy_rate for r in eval_results) / n
        avg_graph_edit_distance = sum(r.graph_edit_distance for r in eval_results) / n
        avg_bertscore = sum(r.bertscore for r in eval_results) / n
        
        # Total and normalized Graph Edit Distance
        total_graph_edit_distance = sum(r.graph_edit_distance for r in eval_results)
        # Normalize by total gold relations (average edits per gold relation)
        normalized_graph_edit_distance = total_graph_edit_distance / total_gold if total_gold > 0 else 0.0
        
        # Calculate partial match statistics (fuzzy matches)
        total_partial_matches = sum(len(r.partial_matches) for r in eval_results)
        avg_partial_matches = total_partial_matches / n if n > 0 else 0.0
        
        # Fuzzy macro averages (average of per-document fuzzy metrics)
        fuzzy_macro_precision = sum(r.fuzzy_precision for r in eval_results) / n
        fuzzy_macro_recall = sum(r.fuzzy_recall for r in eval_results) / n
        fuzzy_macro_f1 = sum(r.fuzzy_f1 for r in eval_results) / n
        
        # Fuzzy micro averages (calculated from aggregated TP/FP/FN including partial matches)
        # Fuzzy TP = exact TP + partial matches
        fuzzy_tp = total_tp + total_partial_matches
        # For fuzzy FP, exclude partial matches from false positives
        fuzzy_fp = total_fp - total_partial_matches
        fuzzy_micro_precision = fuzzy_tp / (fuzzy_tp + fuzzy_fp) if (fuzzy_tp + fuzzy_fp) > 0 else 0.0
        fuzzy_micro_recall = fuzzy_tp / (fuzzy_tp + total_fn) if (fuzzy_tp + total_fn) > 0 else 0.0
        fuzzy_micro_f1 = (
            2 * (fuzzy_micro_precision * fuzzy_micro_recall) / (fuzzy_micro_precision + fuzzy_micro_recall)
            if (fuzzy_micro_precision + fuzzy_micro_recall) > 0 else 0.0
        )
        
        # Statistical spread metrics for key metrics
        # Extract per-document values for statistical calculations
        f1_scores = [r.f1_score for r in eval_results]
        precision_scores = [r.precision for r in eval_results]
        recall_scores = [r.recall for r in eval_results]
        
        # F1 statistics
        f1_std = statistics.stdev(f1_scores) if n > 1 else 0.0
        f1_median = statistics.median(f1_scores)
        f1_min = min(f1_scores)
        f1_max = max(f1_scores)
        
        # Precision statistics
        precision_std = statistics.stdev(precision_scores) if n > 1 else 0.0
        precision_median = statistics.median(precision_scores)
        precision_min = min(precision_scores)
        precision_max = max(precision_scores)
        
        # Recall statistics
        recall_std = statistics.stdev(recall_scores) if n > 1 else 0.0
        recall_median = statistics.median(recall_scores)
        recall_min = min(recall_scores)
        recall_max = max(recall_scores)
        
        return AggregateResults(
            technique_name=technique_name,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            macro_f1=macro_f1,
            micro_precision=micro_precision,
            micro_recall=micro_recall,
            micro_f1=micro_f1,
            avg_exact_match_rate=avg_exact_match_rate,
            avg_omission_rate=avg_omission_rate,
            avg_hallucination_rate=avg_hallucination_rate,
            avg_redundancy_rate=avg_redundancy_rate,
            avg_graph_edit_distance=avg_graph_edit_distance,
            avg_bertscore=avg_bertscore,
            total_tp=total_tp,
            total_fp=total_fp,
            total_fn=total_fn,
            total_gold=total_gold,
            total_predicted=total_predicted,
            overall_exact_match_rate=overall_exact_match_rate,
            overall_omission_rate=overall_omission_rate,
            overall_hallucination_rate=overall_hallucination_rate,
            total_graph_edit_distance=total_graph_edit_distance,
            normalized_graph_edit_distance=normalized_graph_edit_distance,
            total_partial_matches=total_partial_matches,
            avg_partial_matches=avg_partial_matches,
            fuzzy_micro_precision=fuzzy_micro_precision,
            fuzzy_micro_recall=fuzzy_micro_recall,
            fuzzy_micro_f1=fuzzy_micro_f1,
            fuzzy_macro_precision=fuzzy_macro_precision,
            fuzzy_macro_recall=fuzzy_macro_recall,
            fuzzy_macro_f1=fuzzy_macro_f1,
            per_document_results=eval_results,
            # Statistical spread metrics
            f1_std=f1_std,
            f1_median=f1_median,
            f1_min=f1_min,
            f1_max=f1_max,
            precision_std=precision_std,
            precision_median=precision_median,
            precision_min=precision_min,
            precision_max=precision_max,
            recall_std=recall_std,
            recall_median=recall_median,
            recall_min=recall_min,
            recall_max=recall_max,
        )
