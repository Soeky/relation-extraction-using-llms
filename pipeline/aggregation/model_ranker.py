"""Model ranking and summary generation."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

from ..types import AggregateResults


class ModelRanker:
    """Ranks models across all techniques and explains ranking methodology."""
    
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize model ranker.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def rank_models(
        self,
        all_strategy_aggregated: Dict[str, Dict[str, AggregateResults]],
        matching_strategies: List[str],
        output_dir: Path,
        split: str
    ) -> Dict[str, Any]:
        """
        Rank models across all techniques and strategies.
        
        Args:
            all_strategy_aggregated: Dict mapping strategy -> combo_name -> AggregateResults
            matching_strategies: List of matching strategies tested
            output_dir: Output directory for ranking report
            split: Data split name
            
        Returns:
            Ranking report dictionary
        """
        # Extract model performance across all strategies and techniques
        model_performance: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        for strategy in matching_strategies:
            strategy_results = all_strategy_aggregated.get(strategy, {})
            for combo_name, aggregated in strategy_results.items():
                # Parse combo_name: "Technique_Model" or "Technique_Model_Name"
                parts = combo_name.split("_", 1)
                if len(parts) >= 2:
                    technique = parts[0]
                    model_part = parts[1]
                    
                    # Reconstruct model name from combo_name format
                    # combo_name format: technique_model where model has '/' and '-' replaced with '_'
                    model = self._reconstruct_model_name(model_part)
                    
                    # Store performance metrics
                    model_performance[model].append({
                        "strategy": strategy,
                        "technique": technique,
                        "combo_name": combo_name,
                        "macro_f1": aggregated.macro_f1,
                        "macro_precision": aggregated.macro_precision,
                        "macro_recall": aggregated.macro_recall,
                        "micro_f1": aggregated.micro_f1,
                        "micro_precision": aggregated.micro_precision,
                        "micro_recall": aggregated.micro_recall,
                        "fuzzy_macro_f1": aggregated.fuzzy_macro_f1,
                        "fuzzy_micro_f1": aggregated.fuzzy_micro_f1,
                        "avg_exact_match_rate": aggregated.avg_exact_match_rate,
                        "avg_omission_rate": aggregated.avg_omission_rate,
                        "avg_hallucination_rate": aggregated.avg_hallucination_rate,
                        "avg_redundancy_rate": aggregated.avg_redundancy_rate,
                        "avg_graph_edit_distance": aggregated.avg_graph_edit_distance,
                        "total_graph_edit_distance": aggregated.total_graph_edit_distance,
                        "normalized_graph_edit_distance": aggregated.normalized_graph_edit_distance,
                        "num_documents": len(aggregated.per_document_results),
                    })
        
        # Calculate aggregate scores for each model
        model_scores = {}
        for model, performances in model_performance.items():
            model_scores[model] = self._calculate_model_score(model, performances)
        
        # Rank models by overall score
        ranked_models = sorted(
            model_scores.items(),
            key=lambda x: x[1]["overall_score"],
            reverse=True
        )
        
        # Build ranking report
        ranking_report = {
            "methodology": self._get_ranking_methodology(),
            "rankings": [],
            "model_details": {}
        }
        
        # Build detailed rankings
        for rank, (model, score_data) in enumerate(ranked_models, start=1):
            ranking_entry = {
                "rank": rank,
                "model": model,
                "overall_score": score_data["overall_score"],
                "num_techniques": score_data["num_techniques"],
                "num_strategies": score_data["num_strategies"],
                "avg_macro_f1": score_data["avg_macro_f1"],
                "avg_macro_precision": score_data["avg_macro_precision"],
                "avg_macro_recall": score_data["avg_macro_recall"],
                "avg_fuzzy_f1": score_data["avg_fuzzy_f1"],
                "avg_exact_match_rate": score_data["avg_exact_match_rate"],
                "avg_omission_rate": score_data["avg_omission_rate"],
                "avg_hallucination_rate": score_data["avg_hallucination_rate"],
                "ranking_reason": score_data["ranking_reason"],
                "key_strengths": score_data["key_strengths"],
                "key_weaknesses": score_data["key_weaknesses"],
                "score_breakdown": score_data["score_breakdown"]
            }
            
            ranking_report["rankings"].append(ranking_entry)
            ranking_report["model_details"][model] = {
                "performances": model_performance[model],
                "score_breakdown": score_data["score_breakdown"]
            }
        
        # Save ranking report
        ranking_path = output_dir / f"model_rankings_{split}.json"
        with open(ranking_path, 'w', encoding='utf-8') as f:
            json.dump(ranking_report, f, indent=2, ensure_ascii=False)
        
        # Print ranking summary
        self._print_ranking_summary(ranking_report)
        
        self.logger.info(f"\nSaved model rankings to: {ranking_path}")
        
        return ranking_report
    
    def _reconstruct_model_name(self, model_part: str) -> str:
        """Reconstruct model name from combo_name format."""
        # Provider prefixes that should have a slash after them
        providers = ["openai", "anthropic", "google", "meta-llama", "mistralai", 
                    "deepseek", "qwen", "perplexity"]
        
        # Check if model starts with a provider prefix
        model = model_part
        for provider in providers:
            if model_part.startswith(provider + "_"):
                # Replace first underscore after provider with slash
                model = model_part.replace(provider + "_", provider + "/", 1)
                # Replace remaining underscores with dashes
                model = model.replace("_", "-")
                break
        else:
            # No provider prefix, just replace underscores with dashes
            model = model_part.replace("_", "-")
        
        return model
    
    def _calculate_model_score(
        self,
        model: str,
        performances: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate overall score for a model based on its performances.
        
        Ranking Methodology:
        1. Primary metric: Average Macro F1 across all strategies and techniques (weight: 40%)
        2. Secondary metric: Average Fuzzy Macro F1 (weight: 20%)
        3. Precision balance: Average Macro Precision (weight: 15%)
        4. Recall balance: Average Macro Recall (weight: 15%)
        5. Quality metrics: Low omission rate and low hallucination rate (weight: 10%)
        
        The model with the highest overall score ranks first.
        """
        if not performances:
            return {
                "overall_score": 0.0,
                "num_techniques": 0,
                "num_strategies": 0,
                "avg_macro_f1": 0.0,
                "avg_macro_precision": 0.0,
                "avg_macro_recall": 0.0,
                "avg_fuzzy_f1": 0.0,
                "avg_exact_match_rate": 0.0,
                "avg_omission_rate": 1.0,
                "avg_hallucination_rate": 1.0,
                "ranking_reason": "No performance data available",
                "key_strengths": [],
                "key_weaknesses": ["No data available"],
                "score_breakdown": {}
            }
        
        # Calculate averages
        num_performances = len(performances)
        avg_macro_f1 = sum(p["macro_f1"] for p in performances) / num_performances
        avg_macro_precision = sum(p["macro_precision"] for p in performances) / num_performances
        avg_macro_recall = sum(p["macro_recall"] for p in performances) / num_performances
        avg_fuzzy_f1 = sum(p["fuzzy_macro_f1"] for p in performances) / num_performances
        avg_exact_match_rate = sum(p["avg_exact_match_rate"] for p in performances) / num_performances
        avg_omission_rate = sum(p["avg_omission_rate"] for p in performances) / num_performances
        avg_hallucination_rate = sum(p["avg_hallucination_rate"] for p in performances) / num_performances
        
        # Get unique techniques and strategies
        techniques = set(p["technique"] for p in performances)
        strategies = set(p["strategy"] for p in performances)
        
        # Calculate score components (all normalized to 0-1)
        primary_score = avg_macro_f1  # Already 0-1
        fuzzy_score = avg_fuzzy_f1  # Already 0-1
        precision_score = avg_macro_precision  # Already 0-1
        recall_score = avg_macro_recall  # Already 0-1
        quality_score = (1.0 - avg_omission_rate) * 0.6 + (1.0 - avg_hallucination_rate) * 0.4  # Low omission and hallucination is good
        
        # Overall score (weighted combination)
        overall_score = (
            primary_score * 0.40 +
            fuzzy_score * 0.20 +
            precision_score * 0.15 +
            recall_score * 0.15 +
            quality_score * 0.10
        )
        
        # Determine key strengths and weaknesses
        key_strengths = []
        key_weaknesses = []
        
        if avg_macro_f1 >= 0.7:
            key_strengths.append(f"High overall F1 score ({avg_macro_f1:.3f})")
        elif avg_macro_f1 < 0.3:
            key_weaknesses.append(f"Low overall F1 score ({avg_macro_f1:.3f})")
        
        if avg_macro_precision >= 0.75:
            key_strengths.append(f"High precision ({avg_macro_precision:.3f}) - few hallucinations")
        elif avg_macro_precision < 0.4:
            key_weaknesses.append(f"Low precision ({avg_macro_precision:.3f}) - many hallucinations")
        
        if avg_macro_recall >= 0.75:
            key_strengths.append(f"High recall ({avg_macro_recall:.3f}) - captures most relations")
        elif avg_macro_recall < 0.4:
            key_weaknesses.append(f"Low recall ({avg_macro_recall:.3f}) - misses many relations")
        
        if avg_omission_rate < 0.3:
            key_strengths.append(f"Low omission rate ({avg_omission_rate:.3f}) - finds most gold relations")
        elif avg_omission_rate > 0.7:
            key_weaknesses.append(f"High omission rate ({avg_omission_rate:.3f}) - misses many gold relations")
        
        if avg_hallucination_rate < 0.2:
            key_strengths.append(f"Low hallucination rate ({avg_hallucination_rate:.3f}) - few false positives")
        elif avg_hallucination_rate > 0.5:
            key_weaknesses.append(f"High hallucination rate ({avg_hallucination_rate:.3f}) - many false positives")
        
        # Ranking reason
        ranking_reason = self._generate_ranking_reason(
            overall_score, avg_macro_f1, avg_macro_precision, avg_macro_recall,
            avg_omission_rate, avg_hallucination_rate
        )
        
        return {
            "overall_score": overall_score,
            "num_techniques": len(techniques),
            "num_strategies": len(strategies),
            "avg_macro_f1": avg_macro_f1,
            "avg_macro_precision": avg_macro_precision,
            "avg_macro_recall": avg_macro_recall,
            "avg_fuzzy_f1": avg_fuzzy_f1,
            "avg_exact_match_rate": avg_exact_match_rate,
            "avg_omission_rate": avg_omission_rate,
            "avg_hallucination_rate": avg_hallucination_rate,
            "ranking_reason": ranking_reason,
            "key_strengths": key_strengths,
            "key_weaknesses": key_weaknesses,
            "score_breakdown": {
                "primary_score_f1": primary_score,
                "fuzzy_score": fuzzy_score,
                "precision_score": precision_score,
                "recall_score": recall_score,
                "quality_score": quality_score,
                "weighted_components": {
                    "primary_f1_40pct": primary_score * 0.40,
                    "fuzzy_f1_20pct": fuzzy_score * 0.20,
                    "precision_15pct": precision_score * 0.15,
                    "recall_15pct": recall_score * 0.15,
                    "quality_10pct": quality_score * 0.10
                }
            }
        }
    
    def _generate_ranking_reason(
        self,
        overall_score: float,
        avg_f1: float,
        avg_precision: float,
        avg_recall: float,
        avg_omission: float,
        avg_hallucination: float
    ) -> str:
        """Generate human-readable explanation for why this model ranks where it does."""
        reasons = []
        
        if overall_score >= 0.8:
            reasons.append("Excellent overall performance with balanced metrics")
        elif overall_score >= 0.6:
            reasons.append("Strong overall performance")
        elif overall_score >= 0.4:
            reasons.append("Moderate performance")
        else:
            reasons.append("Below average performance")
        
        if avg_f1 >= 0.7:
            reasons.append(f"high F1 score ({avg_f1:.3f}) indicating good precision-recall balance")
        else:
            reasons.append(f"moderate F1 score ({avg_f1:.3f})")
        
        if avg_precision > avg_recall + 0.1:
            reasons.append(f"higher precision ({avg_precision:.3f}) than recall ({avg_recall:.3f}), producing fewer false positives")
        elif avg_recall > avg_precision + 0.1:
            reasons.append(f"higher recall ({avg_recall:.3f}) than precision ({avg_precision:.3f}), capturing more relations")
        else:
            reasons.append(f"balanced precision ({avg_precision:.3f}) and recall ({avg_recall:.3f})")
        
        if avg_omission < 0.3:
            reasons.append(f"low omission rate ({avg_omission:.3f}), finding most gold relations")
        
        if avg_hallucination < 0.2:
            reasons.append(f"low hallucination rate ({avg_hallucination:.3f}), producing few false positives")
        elif avg_hallucination > 0.5:
            reasons.append(f"high hallucination rate ({avg_hallucination:.3f}), producing many false positives")
        
        return ". ".join(reasons) + "."
    
    def _get_ranking_methodology(self) -> Dict[str, Any]:
        """Return the ranking methodology explanation."""
        return {
            "description": "Models are ranked based on a weighted combination of multiple performance metrics.",
            "primary_metric": {
                "name": "Average Macro F1 Score",
                "weight": "40%",
                "description": "Primary indicator of overall model performance. Higher F1 indicates better balance between precision and recall.",
                "calculation": "Average of macro F1 scores across all techniques and matching strategies tested"
            },
            "secondary_metrics": [
                {
                    "name": "Average Fuzzy Macro F1 Score",
                    "weight": "20%",
                    "description": "Accounts for partial/semantic matches, showing model's ability to find relations even with slight variations.",
                    "calculation": "Average of fuzzy macro F1 scores across all techniques and matching strategies"
                },
                {
                    "name": "Average Macro Precision",
                    "weight": "15%",
                    "description": "Indicates how many of the predicted relations are correct. Lower hallucination rate.",
                    "calculation": "Average of macro precision scores across all techniques and matching strategies"
                },
                {
                    "name": "Average Macro Recall",
                    "weight": "15%",
                    "description": "Indicates how many of the gold relations were found. Lower omission rate.",
                    "calculation": "Average of macro recall scores across all techniques and matching strategies"
                },
                {
                    "name": "Quality Score",
                    "weight": "10%",
                    "description": "Combined measure of low omission rate (60%) and low hallucination rate (40%). Lower rates are better.",
                    "calculation": "(1 - avg_omission_rate) * 0.6 + (1 - avg_hallucination_rate) * 0.4"
                }
            ],
            "ranking_criteria": {
                "rule": "Models are sorted by overall_score in descending order (highest score = rank 1)",
                "tie_breaking": "If two models have identical overall_score, the model with higher primary F1 score ranks higher"
            },
            "interpretation": {
                "excellent": "Overall score >= 0.8: Excellent performance across all metrics",
                "good": "Overall score 0.6-0.8: Strong performance with minor weaknesses",
                "moderate": "Overall score 0.4-0.6: Acceptable performance with some areas for improvement",
                "poor": "Overall score < 0.4: Below average performance, significant improvements needed"
            }
        }
    
    def _print_ranking_summary(self, ranking_report: Dict[str, Any]) -> None:
        """Print a formatted ranking summary."""
        self.logger.info("\n" + "=" * 100)
        self.logger.info("MODEL RANKINGS - Best to Worst Performance")
        self.logger.info("=" * 100)
        
        # Print methodology
        methodology = ranking_report["methodology"]
        self.logger.info("\nRanking Methodology:")
        self.logger.info("-" * 100)
        self.logger.info(f"Primary Metric: {methodology['primary_metric']['name']} ({methodology['primary_metric']['weight']})")
        self.logger.info(f"  {methodology['primary_metric']['description']}")
        for metric in methodology["secondary_metrics"]:
            self.logger.info(f"{metric['name']} ({metric['weight']}): {metric['description']}")
        
        # Print rankings table
        self.logger.info("\n" + "=" * 100)
        self.logger.info("Model Rankings:")
        self.logger.info("=" * 100)
        
        header = (
            f"{'Rank':<6} {'Model':<30} {'Score':<8} {'F1':<8} {'Prec':<8} {'Rec':<8} "
            f"{'Fuzzy F1':<10} {'Omit':<8} {'Hall':<8}"
        )
        self.logger.info(header)
        self.logger.info("-" * 100)
        
        for entry in ranking_report["rankings"]:
            rank = entry["rank"]
            model = entry["model"]
            score = entry["overall_score"]
            f1 = entry["avg_macro_f1"]
            prec = entry["avg_macro_precision"]
            rec = entry["avg_macro_recall"]
            fuzzy = entry["avg_fuzzy_f1"]
            omit = entry["avg_omission_rate"]
            hall = entry["avg_hallucination_rate"]
            
            row = (
                f"{rank:<6} {model:<30} {score:<8.3f} {f1:<8.3f} {prec:<8.3f} {rec:<8.3f} "
                f"{fuzzy:<10.3f} {omit:<8.3f} {hall:<8.3f}"
            )
            self.logger.info(row)
        
        # Print top 3 with detailed explanations
        self.logger.info("\n" + "=" * 100)
        self.logger.info("TOP 3 MODELS - Detailed Analysis")
        self.logger.info("=" * 100)
        
        for entry in ranking_report["rankings"][:3]:
            rank = entry["rank"]
            model = entry["model"]
            score = entry["overall_score"]
            
            self.logger.info(f"\nRank #{rank}: {model}")
            self.logger.info(f"  Overall Score: {score:.4f}")
            self.logger.info(f"  Tested with {entry['num_techniques']} technique(s) and {entry['num_strategies']} matching strategy/strategies")
            self.logger.info(f"  Average Metrics:")
            self.logger.info(f"    - Macro F1: {entry['avg_macro_f1']:.3f}")
            self.logger.info(f"    - Macro Precision: {entry['avg_macro_precision']:.3f}")
            self.logger.info(f"    - Macro Recall: {entry['avg_macro_recall']:.3f}")
            self.logger.info(f"    - Fuzzy Macro F1: {entry['avg_fuzzy_f1']:.3f}")
            self.logger.info(f"    - Exact Match Rate: {entry['avg_exact_match_rate']:.3f}")
            self.logger.info(f"    - Omission Rate: {entry['avg_omission_rate']:.3f}")
            self.logger.info(f"    - Hallucination Rate: {entry['avg_hallucination_rate']:.3f}")
            
            self.logger.info(f"  Ranking Reason:")
            self.logger.info(f"    {entry['ranking_reason']}")
            
            if entry["key_strengths"]:
                self.logger.info(f"  Key Strengths:")
                for strength in entry["key_strengths"]:
                    self.logger.info(f"    ✓ {strength}")
            
            if entry["key_weaknesses"]:
                self.logger.info(f"  Key Weaknesses:")
                for weakness in entry["key_weaknesses"]:
                    self.logger.info(f"    ✗ {weakness}")
            
            # Show score breakdown
            breakdown = entry.get("score_breakdown", {})
            if breakdown:
                self.logger.info(f"  Score Breakdown:")
                weighted = breakdown.get("weighted_components", {})
                self.logger.info(f"    - Primary F1 (40%): {weighted.get('primary_f1_40pct', 0):.4f}")
                self.logger.info(f"    - Fuzzy F1 (20%): {weighted.get('fuzzy_f1_20pct', 0):.4f}")
                self.logger.info(f"    - Precision (15%): {weighted.get('precision_15pct', 0):.4f}")
                self.logger.info(f"    - Recall (15%): {weighted.get('recall_15pct', 0):.4f}")
                self.logger.info(f"    - Quality (10%): {weighted.get('quality_10pct', 0):.4f}")
        
        self.logger.info("\n" + "=" * 100)

