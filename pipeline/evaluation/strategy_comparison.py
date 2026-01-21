"""Compare results across different matching strategies."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

from ..types import AggregateResults


def compare_matching_strategies(
    all_strategy_aggregated: Dict[str, Dict[str, AggregateResults]],
    matching_strategies: List[str],
    output_dir: Path,
    split: str,
    logger: logging.Logger
) -> None:
    """
    Compare results across all matching strategies.
    
    Args:
        all_strategy_aggregated: Dict mapping strategy -> combo_name -> AggregateResults
        matching_strategies: List of strategy names
        output_dir: Output directory for comparison results
        split: Data split name
        logger: Logger instance
    """
    logger.info("\n" + "=" * 80)
    logger.info("Matching Strategy Comparison")
    logger.info("=" * 80)
    
    # Collect all unique combinations across all strategies
    all_combinations = set()
    for strategy_results in all_strategy_aggregated.values():
        all_combinations.update(strategy_results.keys())
    
    # Build comparison data structure
    comparison_data = {
        "strategies": matching_strategies,
        "combinations": {},
        "strategy_summary": {}
    }
    
    # For each combination, compare across strategies
    for combo_name in sorted(all_combinations):
        combo_data = {
            "combination": combo_name,
            "strategies": {}
        }
        
        for strategy in matching_strategies:
            if combo_name in all_strategy_aggregated.get(strategy, {}):
                aggregated = all_strategy_aggregated[strategy][combo_name]
                combo_data["strategies"][strategy] = {
                    "macro_precision": aggregated.macro_precision,
                    "macro_recall": aggregated.macro_recall,
                    "macro_f1": aggregated.macro_f1,
                    "micro_precision": aggregated.micro_precision,
                    "micro_recall": aggregated.micro_recall,
                    "micro_f1": aggregated.micro_f1,
                    "fuzzy_macro_f1": aggregated.fuzzy_macro_f1,
                    "fuzzy_micro_f1": aggregated.fuzzy_micro_f1,
                    "avg_exact_match_rate": aggregated.avg_exact_match_rate,
                    "avg_omission_rate": aggregated.avg_omission_rate,
                    "avg_hallucination_rate": aggregated.avg_hallucination_rate,
                    "avg_graph_edit_distance": aggregated.avg_graph_edit_distance,
                    "total_graph_edit_distance": aggregated.total_graph_edit_distance,
                    "normalized_graph_edit_distance": aggregated.normalized_graph_edit_distance,
                    "aggregated_counts": {
                        "total_tp": aggregated.total_tp,
                        "total_fp": aggregated.total_fp,
                        "total_fn": aggregated.total_fn,
                        "total_gold": aggregated.total_gold,
                        "total_predicted": aggregated.total_predicted,
                    },
                    "overall_rates": {
                        "overall_exact_match_rate": aggregated.overall_exact_match_rate,
                        "overall_omission_rate": aggregated.overall_omission_rate,
                        "overall_hallucination_rate": aggregated.overall_hallucination_rate,
                    },
                }
        
        comparison_data["combinations"][combo_name] = combo_data
    
    # Calculate strategy-level summaries
    for strategy in matching_strategies:
        strategy_results = all_strategy_aggregated.get(strategy, {})
        if not strategy_results:
            continue
        
        # Average metrics across all combinations for this strategy
        all_f1s = [r.macro_f1 for r in strategy_results.values()]
        all_precisions = [r.macro_precision for r in strategy_results.values()]
        all_recalls = [r.macro_recall for r in strategy_results.values()]
        all_fuzzy_f1s = [r.fuzzy_macro_f1 for r in strategy_results.values()]
        
        comparison_data["strategy_summary"][strategy] = {
            "num_combinations": len(strategy_results),
            "avg_macro_f1": sum(all_f1s) / len(all_f1s) if all_f1s else 0.0,
            "avg_macro_precision": sum(all_precisions) / len(all_precisions) if all_precisions else 0.0,
            "avg_macro_recall": sum(all_recalls) / len(all_recalls) if all_recalls else 0.0,
            "avg_fuzzy_macro_f1": sum(all_fuzzy_f1s) / len(all_fuzzy_f1s) if all_fuzzy_f1s else 0.0,
            "best_macro_f1": max(all_f1s) if all_f1s else 0.0,
            "worst_macro_f1": min(all_f1s) if all_f1s else 0.0,
        }
    
    # Print comparison table
    logger.info("\nStrategy Comparison Summary:")
    logger.info("-" * 100)
    logger.info(f"{'Strategy':<15} {'Combos':<8} {'Avg F1':<10} {'Avg Prec':<10} {'Avg Rec':<10} {'Avg Fuzzy F1':<12} {'Best F1':<10} {'Worst F1':<10}")
    logger.info("-" * 100)
    
    for strategy in matching_strategies:
        summary = comparison_data["strategy_summary"].get(strategy, {})
        logger.info(
            f"{strategy:<15} {summary.get('num_combinations', 0):<8} "
            f"{summary.get('avg_macro_f1', 0.0):<10.3f} "
            f"{summary.get('avg_macro_precision', 0.0):<10.3f} "
            f"{summary.get('avg_macro_recall', 0.0):<10.3f} "
            f"{summary.get('avg_fuzzy_macro_f1', 0.0):<12.3f} "
            f"{summary.get('best_macro_f1', 0.0):<10.3f} "
            f"{summary.get('worst_macro_f1', 0.0):<10.3f}"
        )
    
    # Find best strategy
    best_strategy = max(
        matching_strategies,
        key=lambda s: comparison_data["strategy_summary"].get(s, {}).get("avg_macro_f1", 0.0)
    )
    logger.info(f"\nBest Strategy (by average F1): {best_strategy}")
    logger.info(f"  Average F1: {comparison_data['strategy_summary'][best_strategy]['avg_macro_f1']:.3f}")
    
    # Print per-combination comparison
    logger.info("\n" + "=" * 80)
    logger.info("Per-Combination Strategy Comparison")
    logger.info("=" * 80)
    
    for combo_name in sorted(all_combinations)[:10]:  # Show first 10
        combo_data = comparison_data["combinations"][combo_name]
        logger.info(f"\n{combo_name}:")
        logger.info(f"  {'Strategy':<15} {'F1':<10} {'Precision':<10} {'Recall':<10} {'Fuzzy F1':<12}")
        logger.info("  " + "-" * 60)
        
        for strategy in matching_strategies:
            if strategy in combo_data["strategies"]:
                metrics = combo_data["strategies"][strategy]
                logger.info(
                    f"  {strategy:<15} "
                    f"{metrics['macro_f1']:<10.3f} "
                    f"{metrics['macro_precision']:<10.3f} "
                    f"{metrics['macro_recall']:<10.3f} "
                    f"{metrics['fuzzy_macro_f1']:<12.3f}"
                )
    
    # Save comparison report
    comparison_path = output_dir / f"strategy_comparison_{split}.json"
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2)
    logger.info(f"\nSaved strategy comparison to: {comparison_path}")
    
    # Identify winners for all metrics
    winners = identify_winners(comparison_data, logger)
    
    # Save winners report
    winners_path = output_dir / f"winners_{split}.json"
    with open(winners_path, 'w', encoding='utf-8') as f:
        json.dump(winners, f, indent=2)
    logger.info(f"Saved winners report to: {winners_path}")
    
    # Print winners summary
    print_winners_summary(winners, logger)


def identify_winners(comparison_data: Dict, logger: logging.Logger) -> Dict:
    """
    Identify winners (best performing combinations) for each metric across all strategies.
    
    Args:
        comparison_data: Strategy comparison data structure
        logger: Logger instance
        
    Returns:
        Dictionary mapping metric names to winners per strategy
    """
    strategies = comparison_data.get("strategies", [])
    combinations = comparison_data.get("combinations", {})
    
    # Define all metrics to find winners for
    metrics = {
        # Core metrics
        "macro_f1": {"higher_is_better": True, "label": "Macro F1"},
        "macro_precision": {"higher_is_better": True, "label": "Macro Precision"},
        "macro_recall": {"higher_is_better": True, "label": "Macro Recall"},
        "micro_f1": {"higher_is_better": True, "label": "Micro F1"},
        "micro_precision": {"higher_is_better": True, "label": "Micro Precision"},
        "micro_recall": {"higher_is_better": True, "label": "Micro Recall"},
        "fuzzy_macro_f1": {"higher_is_better": True, "label": "Fuzzy Macro F1"},
        "fuzzy_micro_f1": {"higher_is_better": True, "label": "Fuzzy Micro F1"},
        # Error metrics (lower is better for omission/hallucination)
        "avg_exact_match_rate": {"higher_is_better": True, "label": "Avg Exact Match Rate"},
        "avg_omission_rate": {"higher_is_better": False, "label": "Avg Omission Rate"},
        "avg_hallucination_rate": {"higher_is_better": False, "label": "Avg Hallucination Rate"},
        # Graph Edit Distance (lower is better)
        "avg_graph_edit_distance": {"higher_is_better": False, "label": "Avg Graph Edit Distance"},
        "total_graph_edit_distance": {"higher_is_better": False, "label": "Total Graph Edit Distance"},
        "normalized_graph_edit_distance": {"higher_is_better": False, "label": "Normalized GED (per gold relation)"},
        # Aggregated counts (higher is better for TP, lower for FP/FN)
        "total_tp": {"higher_is_better": True, "label": "Total TP"},
        "total_fp": {"higher_is_better": False, "label": "Total FP"},
        "total_fn": {"higher_is_better": False, "label": "Total FN"},
        "total_gold": {"higher_is_better": True, "label": "Total Gold"},
        "total_predicted": {"higher_is_better": True, "label": "Total Predicted"},
        # Overall rates
        "overall_exact_match_rate": {"higher_is_better": True, "label": "Overall Exact Match Rate"},
        "overall_omission_rate": {"higher_is_better": False, "label": "Overall Omission Rate"},
        "overall_hallucination_rate": {"higher_is_better": False, "label": "Overall Hallucination Rate"},
    }
    
    winners = {
        "by_strategy": {},
        "by_metric": {},
        "summary": {}
    }
    
    # For each strategy, find winners for each metric
    for strategy in strategies:
        winners["by_strategy"][strategy] = {}
        
        for metric_name, metric_info in metrics.items():
            best_combo = None
            best_value = None
            
            for combo_name, combo_data in combinations.items():
                strategy_data = combo_data.get("strategies", {}).get(strategy, {})
                
                # Handle nested metrics (aggregated_counts, overall_rates)
                if metric_name in ["total_tp", "total_fp", "total_fn", "total_gold", "total_predicted"]:
                    value = strategy_data.get("aggregated_counts", {}).get(metric_name, None)
                elif metric_name.startswith("overall_"):
                    rate_name = metric_name.replace("overall_", "")
                    value = strategy_data.get("overall_rates", {}).get(rate_name, None)
                else:
                    value = strategy_data.get(metric_name, None)
                
                if value is None:
                    continue
                
                if best_value is None:
                    best_value = value
                    best_combo = combo_name
                elif metric_info["higher_is_better"]:
                    if value > best_value:
                        best_value = value
                        best_combo = combo_name
                else:
                    if value < best_value:
                        best_value = value
                        best_combo = combo_name
            
            if best_combo:
                winners["by_strategy"][strategy][metric_name] = {
                    "combination": best_combo,
                    "value": best_value,
                    "label": metric_info["label"]
                }
    
    # For each metric, find overall winner across all strategies
    for metric_name, metric_info in metrics.items():
        best_strategy = None
        best_combo = None
        best_value = None
        
        for strategy in strategies:
            winner_info = winners["by_strategy"].get(strategy, {}).get(metric_name)
            if not winner_info:
                continue
            
            value = winner_info["value"]
            combo = winner_info["combination"]
            
            if best_value is None:
                best_value = value
                best_strategy = strategy
                best_combo = combo
            elif metric_info["higher_is_better"]:
                if value > best_value:
                    best_value = value
                    best_strategy = strategy
                    best_combo = combo
            else:
                if value < best_value:
                    best_value = value
                    best_strategy = strategy
                    best_combo = combo
        
        if best_strategy:
            winners["by_metric"][metric_name] = {
                "strategy": best_strategy,
                "combination": best_combo,
                "value": best_value,
                "label": metric_info["label"]
            }
    
    # Create summary: best combination per strategy (by macro F1)
    for strategy in strategies:
        f1_winner = winners["by_strategy"].get(strategy, {}).get("macro_f1")
        if f1_winner:
            winners["summary"][strategy] = {
                "best_combination": f1_winner["combination"],
                "macro_f1": f1_winner["value"]
            }
    
    return winners


def print_winners_summary(winners: Dict, logger: logging.Logger) -> None:
    """Print a formatted summary of winners."""
    logger.info("\n" + "=" * 80)
    logger.info("WINNERS SUMMARY - Best Combinations for Each Metric")
    logger.info("=" * 80)
    
    strategies = list(winners.get("by_strategy", {}).keys())
    
    # Print winners by strategy
    for strategy in strategies:
        logger.info(f"\n--- {strategy.upper()} Strategy ---")
        strategy_winners = winners["by_strategy"].get(strategy, {})
        
        # Group metrics by category
        core_metrics = ["macro_f1", "macro_precision", "macro_recall", "micro_f1"]
        error_metrics = ["avg_exact_match_rate", "avg_omission_rate", "avg_hallucination_rate"]
        ged_metrics = ["avg_graph_edit_distance", "normalized_graph_edit_distance"]
        aggregated_metrics = ["total_tp", "total_fp", "total_fn", "overall_exact_match_rate"]
        
        logger.info("\n  Core Metrics:")
        for metric in core_metrics:
            if metric in strategy_winners:
                winner = strategy_winners[metric]
                logger.info(f"    {winner['label']:<25} {winner['combination']:<40} {winner['value']:.4f}")
        
        logger.info("\n  Error Metrics:")
        for metric in error_metrics:
            if metric in strategy_winners:
                winner = strategy_winners[metric]
                logger.info(f"    {winner['label']:<25} {winner['combination']:<40} {winner['value']:.4f}")
        
        logger.info("\n  Graph Edit Distance Metrics:")
        for metric in ged_metrics:
            if metric in strategy_winners:
                winner = strategy_winners[metric]
                logger.info(f"    {winner['label']:<25} {winner['combination']:<40} {winner['value']:.4f}")
        
        logger.info("\n  Aggregated Metrics:")
        for metric in aggregated_metrics:
            if metric in strategy_winners:
                winner = strategy_winners[metric]
                logger.info(f"    {winner['label']:<25} {winner['combination']:<40} {winner['value']:.4f}")
    
    # Print overall winners (across all strategies)
    logger.info("\n" + "=" * 80)
    logger.info("OVERALL WINNERS - Best Across All Strategies")
    logger.info("=" * 80)
    
    metric_winners = winners.get("by_metric", {})
    for metric_name, winner in sorted(metric_winners.items()):
        logger.info(f"{winner['label']:<30} {winner['strategy']:<12} {winner['combination']:<40} {winner['value']:.4f}")

