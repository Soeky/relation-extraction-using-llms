"""Technique comparator for comparing different prompting techniques."""

import json
from pathlib import Path
from typing import Dict, List

from ..types import AggregateResults


class TechniqueComparator:
    """Compares different prompting techniques."""
    
    def compare(
        self,
        aggregated_results: Dict[str, AggregateResults]
    ) -> Dict[str, any]:
        """
        Compare aggregated results across techniques.
        
        Args:
            aggregated_results: Dictionary mapping technique names to AggregateResults
            
        Returns:
            Comparison report dictionary
        """
        if not aggregated_results:
            return {}
        
        # Create comparison report
        report = {
            "techniques": {},
            "rankings": {},
            "summary": {}
        }
        
        # Extract metrics for each technique
        for technique_name, results in aggregated_results.items():
            report["techniques"][technique_name] = {
                "macro_precision": results.macro_precision,
                "macro_recall": results.macro_recall,
                "macro_f1": results.macro_f1,
                "micro_precision": results.micro_precision,
                "micro_recall": results.micro_recall,
                "micro_f1": results.micro_f1,
                "avg_exact_match_rate": results.avg_exact_match_rate,
                "avg_omission_rate": results.avg_omission_rate,
                "avg_hallucination_rate": results.avg_hallucination_rate,
                "avg_redundancy_rate": results.avg_redundancy_rate,
                "avg_graph_edit_distance": results.avg_graph_edit_distance,
                "total_graph_edit_distance": results.total_graph_edit_distance,
                "normalized_graph_edit_distance": results.normalized_graph_edit_distance,
                "avg_bertscore": results.avg_bertscore,
                "num_documents": len(results.per_document_results),
                "aggregated_counts": {
                    "total_tp": results.total_tp,
                    "total_fp": results.total_fp,
                    "total_fn": results.total_fn,
                    "total_gold": results.total_gold,
                    "total_predicted": results.total_predicted,
                },
                "overall_rates": {
                    "overall_exact_match_rate": results.overall_exact_match_rate,
                    "overall_omission_rate": results.overall_omission_rate,
                    "overall_hallucination_rate": results.overall_hallucination_rate,
                }
            }
        
        # Create rankings by different metrics
        report["rankings"] = {
            "by_macro_f1": self._rank_by_metric(
                aggregated_results, "macro_f1", reverse=True
            ),
            "by_micro_f1": self._rank_by_metric(
                aggregated_results, "micro_f1", reverse=True
            ),
            "by_precision": self._rank_by_metric(
                aggregated_results, "macro_precision", reverse=True
            ),
            "by_recall": self._rank_by_metric(
                aggregated_results, "macro_recall", reverse=True
            ),
            "by_exact_match_rate": self._rank_by_metric(
                aggregated_results, "avg_exact_match_rate", reverse=True
            ),
            "by_lowest_hallucination": self._rank_by_metric(
                aggregated_results, "avg_hallucination_rate", reverse=False
            ),
            "by_lowest_omission": self._rank_by_metric(
                aggregated_results, "avg_omission_rate", reverse=False
            ),
        }
        
        # Summary statistics
        report["summary"] = {
            "best_overall_f1": max(
                aggregated_results.items(),
                key=lambda x: x[1].macro_f1
            )[0],
            "best_precision": max(
                aggregated_results.items(),
                key=lambda x: x[1].macro_precision
            )[0],
            "best_recall": max(
                aggregated_results.items(),
                key=lambda x: x[1].macro_recall
            )[0],
            "lowest_hallucination": min(
                aggregated_results.items(),
                key=lambda x: x[1].avg_hallucination_rate
            )[0],
            "lowest_omission": min(
                aggregated_results.items(),
                key=lambda x: x[1].avg_omission_rate
            )[0],
        }
        
        return report
    
    def _rank_by_metric(
        self,
        aggregated_results: Dict[str, AggregateResults],
        metric_name: str,
        reverse: bool = True
    ) -> List[tuple]:
        """
        Rank techniques by a specific metric.
        
        Args:
            aggregated_results: Dictionary of results
            metric_name: Name of metric to rank by
            reverse: Whether to sort in descending order
            
        Returns:
            List of (technique_name, metric_value) tuples, sorted
        """
        rankings = []
        for technique_name, results in aggregated_results.items():
            metric_value = getattr(results, metric_name, 0.0)
            rankings.append((technique_name, metric_value))
        
        rankings.sort(key=lambda x: x[1], reverse=reverse)
        return rankings
    
    def save_report(
        self,
        aggregated_results: Dict[str, AggregateResults],
        file_path: str
    ) -> None:
        """
        Save comparison report to JSON file.
        
        Args:
            aggregated_results: Dictionary of aggregated results
            file_path: Path to save report
        """
        report = self.compare(aggregated_results)
        
        # Convert AggregateResults to dict for JSON serialization
        report["techniques"] = {
            name: {
                "macro_precision": float(results.macro_precision),
                "macro_recall": float(results.macro_recall),
                "macro_f1": float(results.macro_f1),
                "micro_precision": float(results.micro_precision),
                "micro_recall": float(results.micro_recall),
                "micro_f1": float(results.micro_f1),
                "avg_exact_match_rate": float(results.avg_exact_match_rate),
                "avg_omission_rate": float(results.avg_omission_rate),
                "avg_hallucination_rate": float(results.avg_hallucination_rate),
                "avg_redundancy_rate": float(results.avg_redundancy_rate),
                "avg_graph_edit_distance": float(results.avg_graph_edit_distance),
                "avg_bertscore": float(results.avg_bertscore),
                "num_documents": len(results.per_document_results),
                "aggregated_counts": {
                    "total_tp": results.total_tp,
                    "total_fp": results.total_fp,
                    "total_fn": results.total_fn,
                    "total_gold": results.total_gold,
                    "total_predicted": results.total_predicted,
                },
                "overall_rates": {
                    "overall_exact_match_rate": float(results.overall_exact_match_rate),
                    "overall_omission_rate": float(results.overall_omission_rate),
                    "overall_hallucination_rate": float(results.overall_hallucination_rate),
                }
            }
            for name, results in aggregated_results.items()
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
    
    def print_comparison_table(
        self,
        aggregated_results: Dict[str, AggregateResults]
    ) -> None:
        """
        Print a formatted comparison table.
        
        Args:
            aggregated_results: Dictionary of aggregated results
        """
        if not aggregated_results:
            print("No results to compare.")
            return
        
        # Header
        print("\n" + "=" * 100)
        print("TECHNIQUE COMPARISON")
        print("=" * 100)
        
        # Table header
        header = f"{'Technique':<12} {'Macro F1':<10} {'Precision':<10} {'Recall':<10} "
        header += f"{'Exact Match':<12} {'Halluc.':<10} {'Omission':<10}"
        print(header)
        print("-" * 100)
        
        # Sort by macro F1
        sorted_results = sorted(
            aggregated_results.items(),
            key=lambda x: x[1].macro_f1,
            reverse=True
        )
        
        # Print each technique
        for technique_name, results in sorted_results:
            row = f"{technique_name:<12} "
            row += f"{results.macro_f1:<10.3f} "
            row += f"{results.macro_precision:<10.3f} "
            row += f"{results.macro_recall:<10.3f} "
            row += f"{results.avg_exact_match_rate:<12.3f} "
            row += f"{results.avg_hallucination_rate:<10.3f} "
            row += f"{results.avg_omission_rate:<10.3f}"
            print(row)
        
        print("=" * 100)
