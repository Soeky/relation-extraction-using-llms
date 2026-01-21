"""Generate bar charts from evaluation results for easy visualization and presentation."""

import json
from pathlib import Path
from typing import Dict, Optional, List

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
    
    # Configure larger default font sizes for better readability
    plt.rcParams.update({
        'font.size': 14,           # Default font size
        'axes.titlesize': 18,      # Title font size
        'axes.labelsize': 16,      # Axis label font size
        'xtick.labelsize': 13,     # X-axis tick label size
        'ytick.labelsize': 13,     # Y-axis tick label size
        'legend.fontsize': 13,     # Legend font size
        'figure.titlesize': 20,    # Figure title size
    })
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# Comprehensive color mapping for all matching strategies
STRATEGY_COLORS = {
    'exact': '#3498db',           # Blue
    'fuzzy': '#2ecc71',           # Green
    'text': '#e74c3c',            # Red
    'bertscore': '#f39c12',       # Orange
    'jaccard': '#9b59b6',         # Purple
    'token': '#1abc9c',           # Turquoise
    'levenshtein': '#e67e22',     # Dark orange
    'jaro_winkler': '#34495e',    # Dark gray
    'sbert': '#16a085',           # Dark turquoise
    'ensemble': '#c0392b',        # Dark red
}


def _get_strategy_color(strategy: str) -> str:
    """Get color for a strategy, with fallback for unknown strategies."""
    return STRATEGY_COLORS.get(strategy, '#95a5a6')  # Gray fallback


def _format_technique_name(technique: str) -> str:
    """Format technique name for display (e.g., 'IO' -> 'I/O')."""
    # Handle prefixes (Baseline-, Improved-)
    if technique.startswith('Baseline-'):
        base = technique.replace('Baseline-', '')
        if base == 'IO':
            return 'Baseline-I/O'
        return f'Baseline-{base}'
    elif technique.startswith('Improved-'):
        base = technique.replace('Improved-', '')
        if base == 'IO':
            return 'Improved-I/O'
        return f'Improved-{base}'
    else:
        # No prefix
        if technique == 'IO':
            return 'I/O'
        return technique


def _calculate_bar_width(num_strategies: int, base_width: float = 0.2) -> float:
    """Calculate appropriate bar width based on number of strategies."""
    # Scale width inversely with number of strategies
    # But don't make bars too narrow (minimum 0.08)
    if num_strategies <= 4:
        return base_width
    elif num_strategies <= 6:
        return 0.15
    elif num_strategies <= 9:
        return 0.12
    else:
        return max(0.08, 0.2 / (num_strategies / 4))


# ============================================================================
# Helper functions for technique extraction and data grouping
# ============================================================================

def _extract_technique(combo_name: str) -> str:
    """Extract prompting technique from combination name.
    
    Args:
        combo_name: Combination name in format "Technique_Model" (e.g., "Baseline-CoT_claude_sonnet_4.5")
        
    Returns:
        Technique name (e.g., "Baseline-CoT") or "Unknown" if format is unexpected
    """
    parts = combo_name.split("_", 1)
    return parts[0] if len(parts) > 1 else "Unknown"


def _extract_model(combo_name: str) -> str:
    """Extract model name from combination name.
    
    Args:
        combo_name: Combination name in format "Technique_Model" (e.g., "Baseline-CoT_claude_sonnet_4.5")
        
    Returns:
        Model name with underscores (e.g., "claude_sonnet_4.5")
    """
    parts = combo_name.split("_", 1)
    return parts[1] if len(parts) > 1 else combo_name


def _reconstruct_model_name(model_part: str) -> str:
    """Reconstruct model name from combo_name format to display format.
    
    Args:
        model_part: Model part from combo_name (e.g., "claude_sonnet_4.5")
        
    Returns:
        Reconstructed model name (e.g., "claude-sonnet-4.5" or "claude/sonnet-4.5" for providers)
    """
    # Provider prefixes that should have a slash after them
    providers = ["openai", "anthropic", "google", "meta-llama", "mistralai", 
                "deepseek", "qwen", "perplexity"]
    
    model = model_part.replace("_", "-")
    
    # Check if model starts with a provider prefix
    for provider in providers:
        if model_part.startswith(provider + "_"):
            # Replace first underscore after provider with slash
            model = model_part.replace(provider + "_", provider + "/", 1)
            # Replace remaining underscores with dashes
            model = model.replace("_", "-")
            return model
    
    # No provider prefix, just replace underscores with dashes
    return model


def _group_by_technique(combinations: Dict) -> Dict[str, Dict]:
    """Group combinations by prompting technique.
    
    Args:
        combinations: Dictionary of combination_name -> combination_data
        
    Returns:
        Dictionary mapping technique -> filtered combinations dict
    """
    grouped = {}
    for combo_name, combo_data in combinations.items():
        technique = _extract_technique(combo_name)
        if technique not in grouped:
            grouped[technique] = {}
        grouped[technique][combo_name] = combo_data
    return grouped


def _filter_by_technique(combinations: Dict, technique: str) -> Dict:
    """Filter combinations to only include those for a specific technique.
    
    Args:
        combinations: Dictionary of combination_name -> combination_data
        technique: Technique name to filter by
        
    Returns:
        Filtered dictionary containing only combinations for the specified technique
    """
    filtered = {}
    for combo_name, combo_data in combinations.items():
        if _extract_technique(combo_name) == technique:
            filtered[combo_name] = combo_data
    return filtered


def _group_rankings_by_technique(rankings: List[Dict]) -> Dict[str, List[Dict]]:
    """Group ranking entries by prompting technique.
    
    Args:
        rankings: List of ranking entries, each with a "model" field
        
    Returns:
        Dictionary mapping technique -> list of ranking entries
    """
    grouped = {}
    for entry in rankings:
        model = entry.get("model", "")
        # Try to extract technique from model name if it contains technique info
        # Otherwise, we need to look at model_details if available
        # For now, we'll need to handle this differently based on data structure
        # This is a placeholder - actual implementation depends on ranking structure
        technique = "Unknown"
        
        # Check if model name contains technique info (unlikely in rankings)
        # We'll need to use model_details or other data to determine technique
        if technique not in grouped:
            grouped[technique] = []
        grouped[technique].append(entry)
    return grouped


def _group_comparisons_by_technique(comparison: Dict) -> Dict[str, Dict]:
    """Group comparison data by prompting technique.
    
    Args:
        comparison: Comparison dictionary with "combinations" key
        
    Returns:
        Dictionary mapping technique -> filtered comparison dict
    """
    combinations = comparison.get("combinations", {})
    strategies = comparison.get("strategies", [])
    strategy_summary = comparison.get("strategy_summary", {})
    
    grouped = {}
    technique_combos = _group_by_technique(combinations)
    
    for technique, filtered_combos in technique_combos.items():
        # Rebuild strategy_summary for this technique
        tech_strategy_summary = {}
        for strategy in strategies:
            strategy_f1s = []
            for combo_name, combo_data in filtered_combos.items():
                strategy_data = combo_data.get("strategies", {}).get(strategy, {})
                f1 = strategy_data.get("macro_f1", 0.0)
                if f1 > 0:
                    strategy_f1s.append(f1)
            
            if strategy_f1s:
                tech_strategy_summary[strategy] = {
                    "avg_macro_f1": np.mean(strategy_f1s) if strategy_f1s else 0.0,
                    "count": len(strategy_f1s)
                }
        
        grouped[technique] = {
            "combinations": filtered_combos,
            "strategies": strategies,
            "strategy_summary": tech_strategy_summary
        }
    
    return grouped


def _aggregate_across_techniques(technique_data: Dict[str, Dict]) -> Dict:
    """Aggregate data across all techniques.
    
    Args:
        technique_data: Dictionary mapping technique -> comparison dict
        
    Returns:
        Aggregated comparison dict with all combinations
    """
    all_combinations = {}
    all_strategies = set()
    
    for technique, comp_data in technique_data.items():
        combinations = comp_data.get("combinations", {})
        strategies = comp_data.get("strategies", [])
        
        all_combinations.update(combinations)
        all_strategies.update(strategies)
    
    # Rebuild strategy_summary across all techniques
    strategy_summary = {}
    for strategy in all_strategies:
        strategy_f1s = []
        for combo_name, combo_data in all_combinations.items():
            strategy_data = combo_data.get("strategies", {}).get(strategy, {})
            f1 = strategy_data.get("macro_f1", 0.0)
            if f1 > 0:
                strategy_f1s.append(f1)
        
        if strategy_f1s:
            strategy_summary[strategy] = {
                "avg_macro_f1": np.mean(strategy_f1s) if strategy_f1s else 0.0,
                "count": len(strategy_f1s)
            }
    
    return {
        "combinations": all_combinations,
        "strategies": sorted(list(all_strategies)),
        "strategy_summary": strategy_summary
    }


def generate_all_charts(
    run_dir: Path,
    split: str,
    exact_rankings: Optional[Dict] = None,
    entity_only_rankings: Optional[Dict] = None,
    exact_comparison: Optional[Dict] = None,
    entity_only_comparison: Optional[Dict] = None,
    logger=None
) -> bool:
    """
    Generate all evaluation charts with automatic directory structure creation.
    
    This function automatically creates the following directory structure:
        run_dir/charts/
            ├── rankings/          - Model ranking charts
            ├── strategies/        - Strategy comparison and analysis charts
            ├── metrics/           - Performance metrics and aggregated counts
            ├── winners/           - Winner charts for each evaluation type
            └── comparisons/       - Exact vs entity-only comparison charts
    
    All directories are created automatically if they don't exist.
    
    Args:
        run_dir: Run directory containing evaluation results
        split: Data split name (e.g., "dev", "test")
        exact_rankings: Optional exact rankings dict (will load from file if None)
        entity_only_rankings: Optional entity-only rankings dict (will load from file if None)
        exact_comparison: Optional exact comparison dict (will load from file if None)
        entity_only_comparison: Optional entity-only comparison dict (will load from file if None)
        logger: Optional logger instance
        
    Returns:
        True if charts were generated successfully, False otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        if logger:
            logger.warning("matplotlib not available, skipping chart generation")
        return False
    
    log_msg = logger.info if logger else print
    
    # Create charts directory - parents=True ensures all parent directories exist
    charts_dir = run_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data if not provided
    if exact_rankings is None:
        exact_file = run_dir / f"model_rankings_{split}.json"
        if exact_file.exists():
            with open(exact_file, 'r', encoding='utf-8') as f:
                exact_rankings = json.load(f)
    
    if entity_only_rankings is None:
        entity_only_file = run_dir / f"model_rankings_{split}_entity_only.json"
        if entity_only_file.exists():
            with open(entity_only_file, 'r', encoding='utf-8') as f:
                entity_only_rankings = json.load(f)
    
    if exact_comparison is None:
        exact_file = run_dir / f"strategy_comparison_{split}.json"
        if exact_file.exists():
            with open(exact_file, 'r', encoding='utf-8') as f:
                exact_comparison = json.load(f)
    
    if entity_only_comparison is None:
        entity_only_file = run_dir / f"strategy_comparison_{split}_entity_only.json"
        if entity_only_file.exists():
            with open(entity_only_file, 'r', encoding='utf-8') as f:
                entity_only_comparison = json.load(f)
    
    if not exact_rankings:
        log_msg("No rankings found, skipping chart generation")
        return False
    
    log_msg("\n" + "=" * 80)
    log_msg("Generating evaluation charts...")
    log_msg("=" * 80)
    
    # Create subdirectories for organized chart storage
    # Using parents=True ensures all parent directories are created automatically
    rankings_dir = charts_dir / "rankings"
    strategies_dir = charts_dir / "strategies"
    metrics_dir = charts_dir / "metrics"
    winners_dir = charts_dir / "winners"
    comparisons_dir = charts_dir / "comparisons"
    
    rankings_dir.mkdir(parents=True, exist_ok=True)
    strategies_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    winners_dir.mkdir(parents=True, exist_ok=True)
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    
    charts_generated = []
    
    try:
        # Generate model ranking charts
        if exact_rankings:
            _create_model_ranking_chart(
                exact_rankings,
                entity_only_rankings,
                rankings_dir / f"model_rankings_exact_{split}.png"
            )
            charts_generated.append(f"rankings/model_rankings_exact_{split}.png")
        
        if entity_only_rankings:
            _create_entity_only_ranking_chart(
                entity_only_rankings,
                rankings_dir / f"model_rankings_entity_only_{split}.png"
            )
            charts_generated.append(f"rankings/model_rankings_entity_only_{split}.png")
            
            _create_comparison_chart(
                exact_rankings,
                entity_only_rankings,
                comparisons_dir / f"comparison_exact_vs_entity_only_{split}.png"
            )
            charts_generated.append(f"comparisons/comparison_exact_vs_entity_only_{split}.png")
            
            _create_metrics_comparison_chart(
                exact_rankings,
                entity_only_rankings,
                comparisons_dir / f"metrics_comparison_{split}.png"
            )
            charts_generated.append(f"comparisons/metrics_comparison_{split}.png")
            
            _create_improvement_chart(
                exact_rankings,
                entity_only_rankings,
                comparisons_dir / f"improvement_chart_{split}.png"
            )
            charts_generated.append(f"comparisons/improvement_chart_{split}.png")
        
        if exact_comparison:
            # Strategy comparison chart (exact mode - shows both modes)
            _create_strategy_comparison_chart(
                exact_comparison,
                entity_only_comparison,
                strategies_dir / f"strategy_comparison_exact_{split}.png"
            )
            charts_generated.append(f"strategies/strategy_comparison_exact_{split}.png")
            
            # New strategy analysis charts
            # Strategy analysis charts (exact mode - with types)
            _create_strategy_model_heatmap(
                exact_comparison,
                strategies_dir / f"strategy_model_heatmap_exact_{split}.png"
            )
            charts_generated.append(f"strategies/strategy_model_heatmap_exact_{split}.png")
            
            _create_best_strategy_per_model(
                exact_comparison,
                strategies_dir / f"best_strategy_per_model_exact_{split}.png"
            )
            charts_generated.append(f"strategies/best_strategy_per_model_exact_{split}.png")
            
            _create_strategy_performance_by_model(
                exact_comparison,
                strategies_dir / f"strategy_performance_by_model_exact_{split}.png"
            )
            charts_generated.append(f"strategies/strategy_performance_by_model_exact_{split}.png")
            
            _create_strategy_by_technique(
                exact_comparison,
                strategies_dir / f"strategy_by_technique_exact_{split}.png"
            )
            charts_generated.append(f"strategies/strategy_by_technique_exact_{split}.png")
            
            # Aggregated metrics charts (exact mode)
            _create_aggregated_counts_chart(
                exact_comparison,
                metrics_dir / f"aggregated_counts_exact_{split}.png"
            )
            charts_generated.append(f"metrics/aggregated_counts_exact_{split}.png")
            
            _create_overall_rates_chart(
                exact_comparison,
                metrics_dir / f"overall_rates_exact_{split}.png"
            )
            charts_generated.append(f"metrics/overall_rates_exact_{split}.png")
            
            _create_total_counts_comparison(
                exact_comparison,
                metrics_dir / f"total_counts_comparison_exact_{split}.png"
            )
            charts_generated.append(f"metrics/total_counts_comparison_exact_{split}.png")
            
            # Additional strategy comparison charts (exact mode)
            _create_strategy_ranking_chart(
                exact_comparison,
                strategies_dir / f"strategy_ranking_exact_{split}.png"
            )
            charts_generated.append(f"strategies/strategy_ranking_exact_{split}.png")
            
            _create_strategy_precision_recall_comparison(
                exact_comparison,
                strategies_dir / f"strategy_precision_recall_exact_{split}.png"
            )
            charts_generated.append(f"strategies/strategy_precision_recall_exact_{split}.png")
            
            _create_strategy_metrics_radar(
                exact_comparison,
                strategies_dir / f"strategy_metrics_radar_exact_{split}.png"
            )
            charts_generated.append(f"strategies/strategy_metrics_radar_exact_{split}.png")
            
            _create_strategy_correlation_heatmap(
                exact_comparison,
                strategies_dir / f"strategy_correlation_exact_{split}.png"
            )
            charts_generated.append(f"strategies/strategy_correlation_exact_{split}.png")
        
        # Generate all strategy charts for entity-only mode
        if entity_only_comparison:
            _create_strategy_comparison_chart(
                entity_only_comparison,
                None,  # No second mode for entity-only comparison
                strategies_dir / f"strategy_comparison_entity_only_{split}.png"
            )
            charts_generated.append(f"strategies/strategy_comparison_entity_only_{split}.png")
            
            # Strategy analysis charts (entity-only mode)
            _create_strategy_model_heatmap(
                entity_only_comparison,
                strategies_dir / f"strategy_model_heatmap_entity_only_{split}.png"
            )
            charts_generated.append(f"strategies/strategy_model_heatmap_entity_only_{split}.png")
            
            _create_best_strategy_per_model(
                entity_only_comparison,
                strategies_dir / f"best_strategy_per_model_entity_only_{split}.png"
            )
            charts_generated.append(f"strategies/best_strategy_per_model_entity_only_{split}.png")
            
            _create_strategy_performance_by_model(
                entity_only_comparison,
                strategies_dir / f"strategy_performance_by_model_entity_only_{split}.png"
            )
            charts_generated.append(f"strategies/strategy_performance_by_model_entity_only_{split}.png")
            
            _create_strategy_by_technique(
                entity_only_comparison,
                strategies_dir / f"strategy_by_technique_entity_only_{split}.png"
            )
            charts_generated.append(f"strategies/strategy_by_technique_entity_only_{split}.png")
            
            # Aggregated metrics charts (entity-only mode)
            _create_aggregated_counts_chart(
                entity_only_comparison,
                metrics_dir / f"aggregated_counts_entity_only_{split}.png"
            )
            charts_generated.append(f"metrics/aggregated_counts_entity_only_{split}.png")
            
            _create_overall_rates_chart(
                entity_only_comparison,
                metrics_dir / f"overall_rates_entity_only_{split}.png"
            )
            charts_generated.append(f"metrics/overall_rates_entity_only_{split}.png")
            
            _create_total_counts_comparison(
                entity_only_comparison,
                metrics_dir / f"total_counts_comparison_entity_only_{split}.png"
            )
            charts_generated.append(f"metrics/total_counts_comparison_entity_only_{split}.png")
            
            # Additional strategy comparison charts (entity-only mode)
            _create_strategy_ranking_chart(
                entity_only_comparison,
                strategies_dir / f"strategy_ranking_entity_only_{split}.png"
            )
            charts_generated.append(f"strategies/strategy_ranking_entity_only_{split}.png")
            
            _create_strategy_precision_recall_comparison(
                entity_only_comparison,
                strategies_dir / f"strategy_precision_recall_entity_only_{split}.png"
            )
            charts_generated.append(f"strategies/strategy_precision_recall_entity_only_{split}.png")
            
            _create_strategy_metrics_radar(
                entity_only_comparison,
                strategies_dir / f"strategy_metrics_radar_entity_only_{split}.png"
            )
            charts_generated.append(f"strategies/strategy_metrics_radar_entity_only_{split}.png")
            
            _create_strategy_correlation_heatmap(
                entity_only_comparison,
                strategies_dir / f"strategy_correlation_entity_only_{split}.png"
            )
            charts_generated.append(f"strategies/strategy_correlation_entity_only_{split}.png")
            
            # Winners chart
            winners_file = run_dir / f"winners_{split}.json"
            if winners_file.exists():
                with open(winners_file, 'r', encoding='utf-8') as f:
                    winners_data = json.load(f)
                _create_winners_chart(
                    winners_data,
                    winners_dir / f"winners_{split}.png"
                )
                # Winners chart now generates multiple files
                charts_generated.extend([
                    f"winners/winners_{split}.png",  # Overview
                    f"winners/winners_exact_{split}.png",
                    f"winners/winners_fuzzy_{split}.png",
                    f"winners/winners_text_{split}.png",
                    f"winners/winners_bertscore_{split}.png"
                ])
                
                # GED-specific chart
                _create_ged_winners_chart(
                    winners_data,
                    winners_dir / f"ged_winners_{split}.png"
                )
                charts_generated.append(f"winners/ged_winners_{split}.png")
        
        # Comprehensive metric charts across all models (exact mode)
        if exact_rankings:
            _create_all_metrics_charts(
                exact_rankings,
                metrics_dir,
                split,
                suffix="exact"
            )
            charts_generated.extend([
                f"metrics/all_metrics_error_rates_exact_{split}.png",
                f"metrics/all_metrics_performance_exact_{split}.png",
                f"metrics/all_metrics_ged_exact_{split}.png",
                f"metrics/all_metrics_comprehensive_exact_{split}.png"
            ])
        
        # Comprehensive metric charts across all models (entity-only mode)
        if entity_only_rankings:
            _create_all_metrics_charts(
                entity_only_rankings,
                metrics_dir,
                split,
                suffix="entity_only"
            )
            charts_generated.extend([
                f"metrics/all_metrics_error_rates_entity_only_{split}.png",
                f"metrics/all_metrics_performance_entity_only_{split}.png",
                f"metrics/all_metrics_ged_entity_only_{split}.png",
                f"metrics/all_metrics_comprehensive_entity_only_{split}.png"
            ])
        
        # Generate per-technique charts
        if exact_comparison:
            techniques_data = _group_comparisons_by_technique(exact_comparison)
            for technique in sorted(techniques_data.keys()):
                _generate_technique_charts(
                    technique,
                    exact_comparison,
                    entity_only_comparison,
                    exact_rankings,
                    entity_only_rankings,
                    charts_dir,
                    split,
                    charts_generated,
                    run_dir=run_dir
                )
        
        # Generate technique comparison charts
        if exact_comparison or entity_only_comparison:
            _generate_technique_comparison_charts(
                exact_comparison,
                entity_only_comparison,
                charts_dir,
                split,
                charts_generated
            )
        
        # Generate overall aggregated charts
        if exact_comparison or entity_only_comparison:
            _generate_overall_charts(
                exact_comparison,
                entity_only_comparison,
                exact_rankings,
                entity_only_rankings,
                charts_dir,
                split,
                charts_generated
            )
        
        log_msg(f"\nGenerated {len(charts_generated)} charts in: {charts_dir}")
        for chart in charts_generated:
            log_msg(f"  - {chart}")
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"Error generating charts: {e}", exc_info=True)
        else:
            print(f"Error generating charts: {e}")
        return False


def _create_model_ranking_chart(
    exact_rankings: Dict,
    entity_only_rankings: Optional[Dict],
    output_path: Path,
    technique: Optional[str] = None
):
    """Create bar chart comparing model rankings.
    
    Args:
        exact_rankings: Rankings dictionary
        entity_only_rankings: Optional entity-only rankings
        output_path: Path to save chart
        technique: Optional prompting technique name to include in title
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Extract data
    models = []
    exact_scores = []
    exact_f1 = []
    
    for entry in exact_rankings.get("rankings", [])[:15]:  # Top 15 models
        models.append(entry["model"])
        exact_scores.append(entry["overall_score"])
        exact_f1.append(entry.get("avg_macro_f1", 0.0))
    
    # Build title with technique info
    title_prefix = f"Model Rankings: {technique} - " if technique else "Model Rankings: "
    
    # Chart 1: Overall Scores
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    bars1 = ax1.barh(models, exact_scores, color=colors)
    ax1.set_xlabel('Overall Score', fontsize=16, fontweight='bold')
    ax1.set_title(f'{title_prefix}With Relation Type Matching\nOverall Score Comparison', 
                  fontsize=18, fontweight='bold', pad=20)
    ax1.set_xlim(0, max(exact_scores) * 1.1 if exact_scores else 1.0)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.invert_yaxis()  # Top model at top
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars1, exact_scores)):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', ha='left', va='center', fontsize=12)
    
    # Chart 2: F1 Scores
    ax2 = axes[1]
    bars2 = ax2.barh(models, exact_f1, color=colors)
    ax2.set_xlabel('Macro F1 Score', fontsize=16, fontweight='bold')
    ax2.set_title(f'{title_prefix}With Relation Type Matching\nMacro F1 Score Comparison', 
                  fontsize=18, fontweight='bold', pad=20)
    ax2.set_xlim(0, max(exact_f1) * 1.1 if exact_f1 else 1.0)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.invert_yaxis()
    
    for i, (bar, score) in enumerate(zip(bars2, exact_f1)):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', ha='left', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_entity_only_ranking_chart(
    entity_only_rankings: Dict,
    output_path: Path,
    technique: Optional[str] = None
):
    """Create bar chart for entity-only rankings.
    
    Args:
        entity_only_rankings: Entity-only rankings dictionary
        output_path: Path to save chart
        technique: Optional prompting technique name to include in title
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Extract data
    models = []
    entity_scores = []
    entity_f1 = []
    
    for entry in entity_only_rankings.get("rankings", [])[:15]:  # Top 15 models
        models.append(entry["model"])
        entity_scores.append(entry["overall_score"])
        entity_f1.append(entry.get("avg_entity_f1", entry.get("avg_fuzzy_f1", 0.0)))
    
    # Build title with technique info
    title_prefix = f"Model Rankings: {technique} - " if technique else "Model Rankings: "
    
    # Chart 1: Overall Scores
    ax1 = axes[0]
    colors = plt.cm.plasma(np.linspace(0, 1, len(models)))
    bars1 = ax1.barh(models, entity_scores, color=colors)
    ax1.set_xlabel('Overall Score', fontsize=16, fontweight='bold')
    ax1.set_title(f'{title_prefix}Entity-Only Matching (Partial Matches = Correct)\nOverall Score Comparison', 
                  fontsize=18, fontweight='bold', pad=20)
    ax1.set_xlim(0, max(entity_scores) * 1.1 if entity_scores else 1.0)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.invert_yaxis()
    
    for i, (bar, score) in enumerate(zip(bars1, entity_scores)):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', ha='left', va='center', fontsize=12)
    
    # Chart 2: Entity F1 Scores
    ax2 = axes[1]
    bars2 = ax2.barh(models, entity_f1, color=colors)
    ax2.set_xlabel('Entity-Only Macro F1 Score', fontsize=16, fontweight='bold')
    ax2.set_title(f'{title_prefix}Entity-Only Matching\nEntity F1 Score Comparison', 
                  fontsize=18, fontweight='bold', pad=20)
    ax2.set_xlim(0, max(entity_f1) * 1.1 if entity_f1 else 1.0)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.invert_yaxis()
    
    for i, (bar, score) in enumerate(zip(bars2, entity_f1)):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', ha='left', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_comparison_chart(
    exact_rankings: Dict,
    entity_only_rankings: Optional[Dict],
    output_path: Path
):
    """Create side-by-side comparison chart of exact vs entity-only scores."""
    if not entity_only_rankings:
        return
    
    # Get common models (top 10 from exact)
    exact_models = [entry["model"] for entry in exact_rankings.get("rankings", [])[:10]]
    entity_only_dict = {entry["model"]: entry for entry in entity_only_rankings.get("rankings", [])}
    
    models = []
    exact_scores = []
    entity_scores = []
    
    for entry in exact_rankings.get("rankings", [])[:10]:
        model = entry["model"]
        models.append(model)
        exact_scores.append(entry["overall_score"])
        entity_scores.append(entity_only_dict.get(model, {}).get("overall_score", 0.0))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, exact_scores, width, label='With Relation Type Matching', 
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, entity_scores, width, label='Entity-Only Matching', 
                   color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=16, fontweight='bold')
    ax.set_ylabel('Overall Score', fontsize=16, fontweight='bold')
    ax.set_title('Model Performance Comparison With Relation Type vs Entity Only Matching',
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=13)
    ax.legend(fontsize=13)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_metrics_comparison_chart(
    exact_rankings: Dict,
    entity_only_rankings: Optional[Dict],
    output_path: Path
):
    """Create chart comparing key metrics (F1, Precision, Recall) for top models."""
    if not entity_only_rankings:
        return
    
    # Get top 5 models from exact rankings
    top_models = [entry["model"] for entry in exact_rankings.get("rankings", [])[:5]]
    exact_dict = {entry["model"]: entry for entry in exact_rankings.get("rankings", [])}
    entity_only_dict = {entry["model"]: entry for entry in entity_only_rankings.get("rankings", [])}
    
    metrics = ['F1', 'Precision', 'Recall']
    exact_values = []
    entity_only_values = []
    
    for model in top_models:
        exact_entry = exact_dict.get(model, {})
        entity_entry = entity_only_dict.get(model, {})
        
        exact_values.append([
            exact_entry.get("avg_macro_f1", 0.0),
            exact_entry.get("avg_macro_precision", 0.0),
            exact_entry.get("avg_macro_recall", 0.0)
        ])
        
        entity_only_values.append([
            entity_entry.get("avg_entity_f1", entity_entry.get("avg_fuzzy_f1", 0.0)),
            entity_entry.get("avg_entity_precision", entity_entry.get("avg_fuzzy_f1", 0.0)),
            entity_entry.get("avg_entity_recall", entity_entry.get("avg_fuzzy_f1", 0.0))
        ])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    x = np.arange(len(top_models))
    width = 0.35
    
    for idx, (metric, ax) in enumerate(zip(metrics, axes)):
        exact_metric = [v[idx] for v in exact_values]
        entity_metric = [v[idx] for v in entity_only_values]
        
        bars1 = ax.bar(x - width/2, exact_metric, width, label='With Relation Type', 
                      color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, entity_metric, width, label='Entity-Only', 
                      color='#e74c3c', alpha=0.8)
        
        ax.set_ylabel(f'{metric} Score', fontsize=15, fontweight='bold')
        ax.set_title(f'{metric} Score Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(top_models, rotation=45, ha='right', fontsize=12)
        ax.set_ylim(0, max(max(exact_metric), max(entity_metric)) * 1.15 if (exact_metric and entity_metric) else 1.0)
        ax.legend(fontsize=13)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=11)
    
    plt.suptitle('Key Metrics Comparison: With Relation Type vs Entity-Only Matching (Top 5 Models)', 
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_improvement_chart(
    exact_rankings: Dict,
    entity_only_rankings: Optional[Dict],
    output_path: Path
):
    """Create chart showing improvement from exact to entity-only matching."""
    if not entity_only_rankings:
        return
    
    # Get top 10 models
    top_models = [entry["model"] for entry in exact_rankings.get("rankings", [])[:10]]
    exact_dict = {entry["model"]: entry for entry in exact_rankings.get("rankings", [])}
    entity_only_dict = {entry["model"]: entry for entry in entity_only_rankings.get("rankings", [])}
    
    models = []
    improvements = []
    
    for model in top_models:
        exact_score = exact_dict.get(model, {}).get("overall_score", 0.0)
        entity_score = entity_only_dict.get(model, {}).get("overall_score", 0.0)
        
        if exact_score > 0:
            improvement = ((entity_score - exact_score) / exact_score) * 100
            models.append(model)
            improvements.append(improvement)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in improvements]
    bars = ax.barh(models, improvements, color=colors, alpha=0.8)
    
    # Set x-axis limits with padding to avoid overlap
    max_improvement = max(improvements) if improvements else 1.0
    min_improvement = min(improvements) if improvements else -1.0
    x_padding = max(max_improvement * 0.1, 5)  # At least 5% padding
    ax.set_xlim(min(min_improvement - x_padding, -x_padding), max_improvement + x_padding)
    
    ax.set_xlabel('Improvement (%)', fontsize=16, fontweight='bold')
    ax.set_title('Performance Improvement: Entity-Only vs With Relation Type Matching\n(Percentage Improvement in Overall Score)', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.tick_params(axis='y', labelsize=14)  # Ensure y-axis labels are readable
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()
    
    # Add value labels with proper spacing to avoid overlap
    for bar, improvement in zip(bars, improvements):
        width = bar.get_width()
        # For 0.0% or very small values, place text slightly to the right to avoid y-axis overlap
        if abs(improvement) < 0.1:
            x_pos = max(0.5, max_improvement * 0.02)  # Small offset from y-axis
            ha = 'left'
        elif improvement > 0:
            x_pos = width + max_improvement * 0.02  # Small padding after bar
            ha = 'left'
        else:
            x_pos = width - max_improvement * 0.02  # Small padding before bar
            ha = 'right'
        
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, 
               f'{improvement:+.1f}%', ha=ha, 
               va='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_strategy_comparison_chart(
    exact_comparison: Dict,
    entity_only_comparison: Optional[Dict],
    output_path: Path
):
    """Create chart comparing different matching strategies."""
    strategies = exact_comparison.get("strategies", [])
    
    if not strategies:
        return
    
    # Extract average F1 scores per strategy
    exact_avg_f1 = []
    entity_avg_f1 = []
    
    strategy_summary = exact_comparison.get("strategy_summary", {})
    for strategy in strategies:
        exact_avg_f1.append(strategy_summary.get(strategy, {}).get("avg_macro_f1", 0.0))
        
        if entity_only_comparison:
            entity_strategy_summary = entity_only_comparison.get("strategy_summary", {})
            entity_avg_f1.append(entity_strategy_summary.get(strategy, {}).get("avg_macro_f1", 0.0))
        else:
            entity_avg_f1.append(0.0)
    
    fig, ax = plt.subplots(figsize=(max(12, len(strategies) * 1.2), 7))
    
    x = np.arange(len(strategies))
    width = 0.35 if not entity_only_comparison else 0.3  # Narrower if both bars
    
    bars1 = ax.bar(x - width/2, exact_avg_f1, width, label='With Type Matching', 
                   color='#3498db', alpha=0.8)
    
    if entity_only_comparison:
        bars2 = ax.bar(x + width/2, entity_avg_f1, width, label='Entity-Only Matching', 
                      color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Matching Strategy', fontsize=16, fontweight='bold')
    ax.set_ylabel('Average Macro F1 Score', fontsize=16, fontweight='bold')
    ax.set_title('Average F1 Score Across All Models',
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    # Rotate labels if many strategies
    label_size = 12 if len(strategies) > 6 else 14
    ax.set_xticklabels(strategies, fontsize=label_size, rotation=45 if len(strategies) > 6 else 0, ha='right' if len(strategies) > 6 else 'center')
    ax.legend(fontsize=14)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=11)
    
    if entity_only_comparison:
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_strategy_model_heatmap(
    exact_comparison: Dict,
    output_path: Path
):
    """Create heatmap showing F1 scores for each strategy-model combination."""
    strategies = exact_comparison.get("strategies", [])
    combinations = exact_comparison.get("combinations", {})
    
    if not strategies or not combinations:
        return
    
    # Extract model names and build matrix
    models = sorted(set(combo.split("_", 1)[1] if "_" in combo else combo 
                       for combo in combinations.keys()))
    
    # Build F1 score matrix: rows = models, cols = strategies
    f1_matrix = []
    model_labels = []
    
    for model in models:
        row = []
        model_labels.append(model)
        for strategy in strategies:
            # Find best F1 for this model-strategy combination
            best_f1 = 0.0
            for combo_name, combo_data in combinations.items():
                if model in combo_name:
                    strategy_data = combo_data.get("strategies", {}).get(strategy, {})
                    f1 = strategy_data.get("macro_f1", 0.0)
                    best_f1 = max(best_f1, f1)
            row.append(best_f1)
        f1_matrix.append(row)
    
    if not f1_matrix:
        return
    
    fig, ax = plt.subplots(figsize=(max(10, len(strategies) * 1.0), max(8, len(models) * 0.5)))
    
    max_val = max(max(row) for row in f1_matrix) if f1_matrix else 1.0
    im = ax.imshow(f1_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=max_val)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(strategies)))
    ax.set_yticks(np.arange(len(model_labels)))
    label_size = 11 if len(strategies) > 6 else 13
    ax.set_xticklabels(strategies, fontsize=label_size)
    ax.set_yticklabels(model_labels, fontsize=12)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations (bold the best result per model row)
    for i in range(len(model_labels)):
        row_max = max(f1_matrix[i]) if f1_matrix[i] else 0  # Find best in this row
        for j in range(len(strategies)):
            is_best = f1_matrix[i][j] == row_max and row_max > 0
            text = ax.text(j, i, f'{f1_matrix[i][j]:.3f}',
                          ha="center", va="center", color="black", fontsize=11,
                          fontweight='bold' if is_best else 'normal')
    
    ax.set_title("Model Performance Heatmap\n(Macro F1 Scores)",
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("Evaluation Method", fontsize=16, fontweight='bold')
    ax.set_ylabel("Model", fontsize=16, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Macro F1 Score', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_best_strategy_per_model(
    exact_comparison: Dict,
    output_path: Path,
    technique: Optional[str] = None
):
    """Create chart showing the best strategy for each model.
    
    Args:
        exact_comparison: Comparison dictionary
        output_path: Path to save chart
        technique: Optional prompting technique name to include in title and labels
    """
    strategies = exact_comparison.get("strategies", [])
    combinations = exact_comparison.get("combinations", {})
    
    if not strategies or not combinations:
        return
    
    # Find best strategy for each model, including technique info
    model_best_strategy = {}
    model_best_f1 = {}
    model_best_technique = {}
    
    for combo_name, combo_data in combinations.items():
        # Extract technique and model name
        technique_name = _extract_technique(combo_name)
        model_part = _extract_model(combo_name)
        model = _reconstruct_model_name(model_part)
        
        best_strategy = None
        best_f1 = -1.0
        
        for strategy in strategies:
            strategy_data = combo_data.get("strategies", {}).get(strategy, {})
            f1 = strategy_data.get("macro_f1", 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_strategy = strategy
        
        if best_strategy:
            # Keep track of best across all combinations for this model
            # If multiple techniques, show the best one
            if model not in model_best_f1 or best_f1 > model_best_f1[model]:
                model_best_strategy[model] = best_strategy
                model_best_f1[model] = best_f1
                model_best_technique[model] = technique_name
    
    if not model_best_strategy:
        return
    
    # Sort by F1 score
    sorted_models = sorted(model_best_strategy.items(), 
                          key=lambda x: model_best_f1[x[0]], reverse=True)
    
    models = [m[0] for m in sorted_models[:20]]  # Top 20 models
    best_strategies = [model_best_strategy[m] for m in models]
    f1_scores = [model_best_f1[m] for m in models]
    best_techniques = [model_best_technique.get(m, "") for m in models]
    
    # Build model labels with technique info if not filtering by technique
    if technique:
        model_labels = models  # Just model name, technique is in title
    else:
        model_labels = [f"{m} ({t})" if t else m for m, t in zip(models, best_techniques)]
    
    # Color by strategy
    colors = [_get_strategy_color(s) for s in best_strategies]
    
    fig, ax = plt.subplots(figsize=(14, max(8, len(models) * 0.4)))
    
    bars = ax.barh(model_labels, f1_scores, color=colors, alpha=0.8)
    
    ax.set_xlabel('Macro F1 Score', fontsize=16, fontweight='bold')
    title = 'Model Performance Ranking'
    if technique:
        title = f'{title}: {technique}'
    title = f'{title}\n(Top 20 Models)'
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_xlim(0, max(f1_scores) * 1.1 if f1_scores else 1.0)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()
    
    # Add value labels
    for bar, score, strategy in zip(bars, f1_scores, best_strategies):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
               f'{score:.3f} ({strategy})', ha='left', va='center', 
               fontsize=11, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=_get_strategy_color(s), 
                            label=s) for s in strategies]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=13)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_strategy_performance_by_model(
    exact_comparison: Dict,
    output_path: Path
):
    """Create grouped bar chart showing all strategies' performance for each model."""
    strategies = exact_comparison.get("strategies", [])
    combinations = exact_comparison.get("combinations", {})
    
    if not strategies or not combinations:
        return
    
    # Get top models (by best F1 across all strategies)
    model_best_f1 = {}
    for combo_name, combo_data in combinations.items():
        parts = combo_name.split("_", 1)
        model = parts[1] if len(parts) > 1 else combo_name
        
        best_f1 = 0.0
        for strategy in strategies:
            strategy_data = combo_data.get("strategies", {}).get(strategy, {})
            f1 = strategy_data.get("macro_f1", 0.0)
            best_f1 = max(best_f1, f1)
        
        if model not in model_best_f1 or best_f1 > model_best_f1[model]:
            model_best_f1[model] = best_f1
    
    # Get top 10 models
    top_models = sorted(model_best_f1.items(), key=lambda x: x[1], reverse=True)[:10]
    models = [m[0] for m in top_models]
    
    # Build data matrix
    strategy_data = {s: [] for s in strategies}
    for model in models:
        for strategy in strategies:
            best_f1 = 0.0
            for combo_name, combo_data in combinations.items():
                if model in combo_name:
                    strategy_data_dict = combo_data.get("strategies", {}).get(strategy, {})
                    f1 = strategy_data_dict.get("macro_f1", 0.0)
                    best_f1 = max(best_f1, f1)
            strategy_data[strategy].append(best_f1)
    
    fig, ax = plt.subplots(figsize=(max(16, len(models) * 1.5), 8))
    
    x = np.arange(len(models))
    width = _calculate_bar_width(len(strategies))
    
    for i, strategy in enumerate(strategies):
        offset = (i - len(strategies)/2 + 0.5) * width
        bars = ax.bar(x + offset, strategy_data[strategy], width, 
                     label=strategy, color=_get_strategy_color(strategy), 
                     alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, strategy_data[strategy]):
            if val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Model', fontsize=16, fontweight='bold')
    ax.set_ylabel('Macro F1 Score', fontsize=16, fontweight='bold')
    ax.set_title('Model Performance Comparison\n(Top 10 Models)',
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=12)
    ax.legend(fontsize=13)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_strategy_by_technique(
    exact_comparison: Dict,
    output_path: Path
):
    """Create chart comparing strategies across different techniques."""
    strategies = exact_comparison.get("strategies", [])
    combinations = exact_comparison.get("combinations", {})
    
    if not strategies or not combinations:
        return
    
    # Extract techniques
    techniques = set()
    for combo_name in combinations.keys():
        parts = combo_name.split("_", 1)
        technique = parts[0] if len(parts) > 1 else "Unknown"
        techniques.add(technique)
    
    techniques = sorted(techniques)
    
    # Build data: technique -> strategy -> avg F1
    technique_strategy_f1 = {t: {s: [] for s in strategies} for t in techniques}
    
    for combo_name, combo_data in combinations.items():
        parts = combo_name.split("_", 1)
        technique = parts[0] if len(parts) > 1 else "Unknown"
        
        for strategy in strategies:
            strategy_data = combo_data.get("strategies", {}).get(strategy, {})
            f1 = strategy_data.get("macro_f1", 0.0)
            technique_strategy_f1[technique][strategy].append(f1)
    
    # Calculate averages
    avg_f1 = {t: {s: np.mean(technique_strategy_f1[t][s]) if technique_strategy_f1[t][s] else 0.0
                  for s in strategies} for t in techniques}
    
    # For many strategies, use a wider figure and smaller bars
    num_strategies = len(strategies)
    fig_width = max(14, len(techniques) * 3.5) if num_strategies > 6 else max(12, len(techniques) * 2)
    fig_height = 8 if num_strategies > 6 else 7
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    x = np.arange(len(techniques))
    # Calculate width more carefully - ensure bars don't overlap too much
    # Each technique group needs space for all strategy bars
    total_width_per_group = 0.8  # Use 80% of the space between techniques
    width = min(0.15, total_width_per_group / num_strategies)  # Max 0.15 per bar
    
    for i, strategy in enumerate(strategies):
        # Center the bars around each technique
        offset = (i - (num_strategies - 1) / 2) * width
        values = [avg_f1[t][strategy] for t in techniques]
        bars = ax.bar(x + offset, values, width, 
                     label=strategy, color=_get_strategy_color(strategy), 
                     alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Add value labels only for top performers or if value > 0.01 to avoid clutter
        for bar, val in zip(bars, values):
            if val > 0.01:
                height = bar.get_height()
                # Only show label if there's room (not too crowded)
                if num_strategies <= 6 or val > 0.05:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Technique', fontsize=16, fontweight='bold')
    ax.set_ylabel('Average Macro F1 Score', fontsize=16, fontweight='bold')
    ax.set_title('Performance by Prompting Technique\n(Average F1 Across All Models)',
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([_format_technique_name(t) for t in techniques], fontsize=13)
    
    # Adjust legend - if too many strategies, use smaller font or 2 columns
    legend_fontsize = 10 if num_strategies > 6 else 13
    ncol = 2 if num_strategies > 6 else 1
    ax.legend(fontsize=legend_fontsize, ncol=ncol, loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_aggregated_counts_chart(
    exact_comparison: Dict,
    output_path: Path
):
    """Create chart showing aggregated counts (TP, FP, FN) across all documents for top combinations."""
    combinations = exact_comparison.get("combinations", {})
    strategies = exact_comparison.get("strategies", [])
    
    if not combinations or not strategies:
        return
    
    # Get top 10 combinations by F1 score
    combo_scores = []
    for combo_name, combo_data in combinations.items():
        best_f1 = 0.0
        for strategy in strategies:
            strategy_data = combo_data.get("strategies", {}).get(strategy, {})
            f1 = strategy_data.get("macro_f1", 0.0)
            best_f1 = max(best_f1, f1)
        combo_scores.append((combo_name, best_f1))
    
    combo_scores.sort(key=lambda x: x[1], reverse=True)
    top_combos = [c[0] for c in combo_scores[:10]]
    
    # Extract counts for best strategy per combination
    combo_labels = []
    tp_counts = []
    fp_counts = []
    fn_counts = []
    gold_counts = []
    predicted_counts = []
    
    for combo_name in top_combos:
        combo_data = combinations.get(combo_name, {})
        
        # Find best strategy for this combo
        best_strategy = None
        best_f1 = -1.0
        for strategy in strategies:
            strategy_data = combo_data.get("strategies", {}).get(strategy, {})
            f1 = strategy_data.get("macro_f1", 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_strategy = strategy
        
        if best_strategy:
            strategy_data = combo_data.get("strategies", {}).get(best_strategy, {})
            counts = strategy_data.get("aggregated_counts", {})
            
            combo_labels.append(f"{combo_name}\n({best_strategy})")
            tp_counts.append(counts.get("total_tp", 0))
            fp_counts.append(counts.get("total_fp", 0))
            fn_counts.append(counts.get("total_fn", 0))
            gold_counts.append(counts.get("total_gold", 0))
            predicted_counts.append(counts.get("total_predicted", 0))
    
    if not combo_labels:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Chart 1: Stacked bar chart of TP, FP, FN
    ax1 = axes[0]
    x = np.arange(len(combo_labels))
    width = 0.6
    
    bars1 = ax1.bar(x, tp_counts, width, label='True Positives (TP)', color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x, fp_counts, width, bottom=tp_counts, label='False Positives (FP)', 
                    color='#e74c3c', alpha=0.8)
    bars3 = ax1.bar(x, fn_counts, width, bottom=np.array(tp_counts) + np.array(fp_counts), 
                    label='False Negatives (FN)', color='#f39c12', alpha=0.8)
    
    ax1.set_xlabel('Combination (Best Strategy)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=16, fontweight='bold')
    ax1.set_title('Aggregated Counts Across All Documents\n(TP, FP, FN for Top 10 Combinations)', 
                  fontsize=18, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(combo_labels, rotation=45, ha='right', fontsize=11)
    ax1.legend(fontsize=13, loc='upper left')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Chart 2: Total Gold vs Total Predicted
    ax2 = axes[1]
    x2 = np.arange(len(combo_labels))
    width2 = 0.35
    
    bars4 = ax2.bar(x2 - width2/2, gold_counts, width2, label='Total Gold Relations', 
                    color='#3498db', alpha=0.8)
    bars5 = ax2.bar(x2 + width2/2, predicted_counts, width2, label='Total Predicted Relations', 
                    color='#9b59b6', alpha=0.8)
    
    ax2.set_xlabel('Combination (Best Strategy)', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=16, fontweight='bold')
    ax2.set_title('Total Gold vs Total Predicted Relations\n(All Documents Combined)', 
                  fontsize=18, fontweight='bold', pad=20)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(combo_labels, rotation=45, ha='right', fontsize=11)
    ax2.legend(fontsize=13)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bars in [bars4, bars5]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_overall_rates_chart(
    exact_comparison: Dict,
    output_path: Path
):
    """Create chart showing overall rates (exact match, omission, hallucination) across all documents."""
    combinations = exact_comparison.get("combinations", {})
    strategies = exact_comparison.get("strategies", [])
    
    if not combinations or not strategies:
        return
    
    # Get top 10 combinations by F1 score
    combo_scores = []
    for combo_name, combo_data in combinations.items():
        best_f1 = 0.0
        for strategy in strategies:
            strategy_data = combo_data.get("strategies", {}).get(strategy, {})
            f1 = strategy_data.get("macro_f1", 0.0)
            best_f1 = max(best_f1, f1)
        combo_scores.append((combo_name, best_f1))
    
    combo_scores.sort(key=lambda x: x[1], reverse=True)
    top_combos = [c[0] for c in combo_scores[:10]]
    
    # Extract overall rates for best strategy per combination
    combo_labels = []
    exact_match_rates = []
    omission_rates = []
    hallucination_rates = []
    
    for combo_name in top_combos:
        combo_data = combinations.get(combo_name, {})
        
        # Find best strategy for this combo
        best_strategy = None
        best_f1 = -1.0
        for strategy in strategies:
            strategy_data = combo_data.get("strategies", {}).get(strategy, {})
            f1 = strategy_data.get("macro_f1", 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_strategy = strategy
        
        if best_strategy:
            strategy_data = combo_data.get("strategies", {}).get(best_strategy, {})
            rates = strategy_data.get("overall_rates", {})
            
            combo_labels.append(f"{combo_name}\n({best_strategy})")
            exact_match_rates.append(rates.get("overall_exact_match_rate", 0.0) * 100)
            omission_rates.append(rates.get("overall_omission_rate", 0.0) * 100)
            hallucination_rates.append(rates.get("overall_hallucination_rate", 0.0) * 100)
    
    if not combo_labels:
        return
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(combo_labels))
    width = 0.25
    
    bars1 = ax.bar(x - width, exact_match_rates, width, label='Overall Exact Match Rate (%)', 
                   color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, omission_rates, width, label='Overall Omission Rate (%)', 
                   color='#f39c12', alpha=0.8)
    bars3 = ax.bar(x + width, hallucination_rates, width, label='Overall Hallucination Rate (%)', 
                   color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Combination (Best Strategy)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Rate (%)', fontsize=16, fontweight='bold')
    ax.set_title('Overall Rates Across All Documents\n(Exact Match, Omission, Hallucination for Top 10 Combinations)', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(combo_labels, rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=13)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_total_counts_comparison(
    exact_comparison: Dict,
    output_path: Path
):
    """Create chart comparing total counts across different strategies."""
    strategies = exact_comparison.get("strategies", [])
    combinations = exact_comparison.get("combinations", {})
    
    if not strategies or not combinations:
        return
    
    # Aggregate counts across all combinations for each strategy
    strategy_totals = {s: {"tp": 0, "fp": 0, "fn": 0, "gold": 0, "predicted": 0} 
                       for s in strategies}
    
    for combo_name, combo_data in combinations.items():
        for strategy in strategies:
            strategy_data = combo_data.get("strategies", {}).get(strategy, {})
            counts = strategy_data.get("aggregated_counts", {})
            
            strategy_totals[strategy]["tp"] += counts.get("total_tp", 0)
            strategy_totals[strategy]["fp"] += counts.get("total_fp", 0)
            strategy_totals[strategy]["fn"] += counts.get("total_fn", 0)
            strategy_totals[strategy]["gold"] += counts.get("total_gold", 0)
            strategy_totals[strategy]["predicted"] += counts.get("total_predicted", 0)
    
    # Normalize by number of combinations to get averages
    num_combos = len(combinations)
    if num_combos == 0:
        return
    
    strategy_avg_tp = [strategy_totals[s]["tp"] / num_combos for s in strategies]
    strategy_avg_fp = [strategy_totals[s]["fp"] / num_combos for s in strategies]
    strategy_avg_fn = [strategy_totals[s]["fn"] / num_combos for s in strategies]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Chart 1: Average TP, FP, FN per strategy
    ax1 = axes[0]
    x = np.arange(len(strategies))
    width = 0.25
    
    bars1 = ax1.bar(x - width, strategy_avg_tp, width, label='Avg TP', 
                    color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x, strategy_avg_fp, width, label='Avg FP', 
                    color='#e74c3c', alpha=0.8)
    bars3 = ax1.bar(x + width, strategy_avg_fn, width, label='Avg FN', 
                    color='#f39c12', alpha=0.8)
    
    ax1.set_xlabel('Matching Strategy', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Average Count (per combination)', fontsize=16, fontweight='bold')
    ax1.set_title('Average Aggregated Counts by Strategy\n(TP, FP, FN averaged across all combinations)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, fontsize=14)
    ax1.legend(fontsize=13)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=11)
    
    # Chart 2: Total counts stacked
    ax2 = axes[1]
    strategy_total_tp = [strategy_totals[s]["tp"] for s in strategies]
    strategy_total_fp = [strategy_totals[s]["fp"] for s in strategies]
    strategy_total_fn = [strategy_totals[s]["fn"] for s in strategies]
    
    bars4 = ax2.bar(strategies, strategy_total_tp, label='Total TP', 
                    color='#2ecc71', alpha=0.8)
    bars5 = ax2.bar(strategies, strategy_total_fp, bottom=strategy_total_tp, 
                    label='Total FP', color='#e74c3c', alpha=0.8)
    bars6 = ax2.bar(strategies, strategy_total_fn, 
                    bottom=np.array(strategy_total_tp) + np.array(strategy_total_fp), 
                    label='Total FN', color='#f39c12', alpha=0.8)
    
    ax2.set_xlabel('Matching Strategy', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Total Count (all combinations)', fontsize=16, fontweight='bold')
    ax2.set_title('Total Aggregated Counts by Strategy\n(Sum across all combinations)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax2.legend(fontsize=13)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_winners_chart(
    winners_data: Dict,
    output_path: Path,
    technique: Optional[str] = None
):
    """Create separate, clear charts for each evaluation type showing winners for micro and macro metrics.
    
    Args:
        winners_data: Winners data dictionary
        output_path: Path to save chart
        technique: Optional prompting technique name to include in title
    """
    by_strategy = winners_data.get("by_strategy", {})
    
    if not by_strategy:
        return
    
    # Get all strategies from data
    strategies = sorted(by_strategy.keys())
    
    # Define key micro and macro metrics to visualize
    key_metrics = [
        ("macro_f1", "Macro F1"),
        ("macro_precision", "Macro Precision"),
        ("macro_recall", "Macro Recall"),
        ("micro_f1", "Micro F1"),
        ("micro_precision", "Micro Precision"),
        ("micro_recall", "Micro Recall"),
    ]
    
    # Create a separate chart for each evaluation type
    for strategy in strategies:
        if strategy not in by_strategy:
            continue
        
        # Create figure for this strategy
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Collect data for all metrics
        metrics_labels = []
        combinations = []
        values = []
        colors = []
        
        for metric_name, metric_label in key_metrics:
            winner_info = by_strategy.get(strategy, {}).get(metric_name)
            if winner_info:
                metrics_labels.append(metric_label)
                combo_name = winner_info["combination"]
                # For per-technique charts, show just model name; otherwise show full combo
                if technique:
                    model_part = _extract_model(combo_name)
                    display_name = _reconstruct_model_name(model_part)
                else:
                    display_name = combo_name
                combinations.append(display_name)
                values.append(winner_info["value"])
                # Use strategy color
                colors.append(_get_strategy_color(strategy))
        
        if not metrics_labels:
            plt.close()
            continue
        
        # Create horizontal bar chart
        y_pos = np.arange(len(metrics_labels))
        bars = ax.barh(y_pos, values, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
        
        # Customize appearance
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics_labels, fontsize=14, fontweight='bold')
        ax.invert_yaxis()  # Top metric at top
        ax.set_xlabel('Score', fontsize=16, fontweight='bold')
        title = f'{strategy.upper()} Evaluation: Best Performers for Each Metric'
        if technique:
            title = f'{technique} - {title}'
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=1)
        
        # Add value labels and model names
        for idx, (bar, value, combo) in enumerate(zip(bars, values, combinations)):
            width = bar.get_width()
            # Show score
            ax.text(width + max(values) * 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{value:.4f}', 
                   ha='left', va='center', fontsize=13, fontweight='bold')
            # Show model name on the bar
            ax.text(width * 0.02, bar.get_y() + bar.get_height()/2, 
                   combo, 
                   ha='left', va='center', fontsize=11, color='white', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        # Add legend to distinguish macro vs micro
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=_get_strategy_color(strategy), alpha=0.85, 
                  edgecolor='black', linewidth=1.5, label='Macro & Micro Metrics')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=12)
        
        # Set x-axis limit with some padding
        ax.set_xlim(0, max(values) * 1.25)
        
        plt.tight_layout()
        
        # Save to separate file
        output_dir = output_path.parent
        strategy_filename = output_path.stem.replace('winners', f'winners_{strategy}') + output_path.suffix
        plt.savefig(output_dir / strategy_filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Also create an overview comparison chart
    _create_winners_overview(winners_data, output_path)


def _create_winners_overview(
    winners_data: Dict,
    output_path: Path
):
    """Create an overview chart comparing F1 scores across all evaluation types."""
    by_strategy = winners_data.get("by_strategy", {})
    
    if not by_strategy:
        return
    
    strategies = sorted(by_strategy.keys())
    available_strategies = [s for s in strategies if s in by_strategy]
    
    if not available_strategies:
        return
    
    # Create figure with 2 subplots: Macro F1 and Micro F1
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    for idx, (metric_name, metric_label) in enumerate([("macro_f1", "Macro F1"), ("micro_f1", "Micro F1")]):
        ax = axes[idx]
        
        strategy_names = []
        combinations = []
        values = []
        colors = []
        
        for strategy in strategies:
            if strategy not in by_strategy:
                continue
            
            winner_info = by_strategy.get(strategy, {}).get(metric_name)
            if winner_info:
                strategy_names.append(strategy.upper())
                combinations.append(winner_info["combination"])
                values.append(winner_info["value"])
                colors.append(_get_strategy_color(strategy))
        
        if not strategy_names:
            continue
        
        # Create horizontal bar chart
        y_pos = np.arange(len(strategy_names))
        bars = ax.barh(y_pos, values, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
        
        # Customize appearance
        ax.set_yticks(y_pos)
        ax.set_yticklabels(strategy_names, fontsize=16, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlabel('F1 Score', fontsize=18, fontweight='bold')
        ax.set_title(f'{metric_label} Winners by Evaluation Type', 
                    fontsize=22, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=1)
        
        # Add value labels and model names
        for bar, value, combo in zip(bars, values, combinations):
            width = bar.get_width()
            # Show score
            ax.text(width + max(values) * 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{value:.4f}', 
                   ha='left', va='center', fontsize=15, fontweight='bold')
            # Show model name
            ax.text(width * 0.5, bar.get_y() + bar.get_height()/2, 
                   combo, 
                   ha='center', va='center', fontsize=13, color='white', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
        
        ax.set_xlim(0, max(values) * 1.3)
    
    plt.suptitle('Winners Overview: F1 Score Comparison Across Evaluation Types', 
                 fontsize=24, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_ged_winners_chart(
    winners_data: Dict,
    output_path: Path
):
    """Create chart showing Graph Edit Distance winners (lowest GED = best)."""
    by_strategy = winners_data.get("by_strategy", {})
    by_metric = winners_data.get("by_metric", {})
    
    if not by_strategy:
        return
    
    strategies = list(by_strategy.keys())
    
    # GED metrics
    ged_metrics = [
        ("avg_graph_edit_distance", "Avg Graph Edit Distance"),
        ("normalized_graph_edit_distance", "Normalized GED (per gold relation)"),
        ("total_graph_edit_distance", "Total Graph Edit Distance"),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for idx, (metric_name, metric_label) in enumerate(ged_metrics):
        ax = axes[idx]
        
        # Get winners for this metric across all strategies
        strategy_winners = []
        combo_names = []
        values = []
        
        for strategy in strategies:
            winner_info = by_strategy.get(strategy, {}).get(metric_name)
            if winner_info:
                strategy_winners.append(strategy)
                combo_names.append(winner_info["combination"])
                values.append(winner_info["value"])
        
        if not strategy_winners:
            ax.text(0.5, 0.5, f"No data for {metric_label}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(metric_label, fontsize=14, fontweight='bold')
            ax.axis('off')
            continue
        
        # Sort by value (lowest first for GED)
        sorted_data = sorted(zip(strategy_winners, combo_names, values), key=lambda x: x[2])
        strategy_winners, combo_names, values = zip(*sorted_data) if sorted_data else ([], [], [])
        
        # Use color gradient: green (low) to red (high)
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(strategy_winners)))
        bars = ax.barh(strategy_winners, values, color=colors, alpha=0.8)
        
        # Add value labels
        for bar, value, combo in zip(bars, values, combo_names):
            width = bar.get_width()
            # Truncate combo name for display
            combo_short = combo[:35] + "..." if len(combo) > 35 else combo
            if metric_name == "normalized_graph_edit_distance":
                value_str = f'{value:.3f}\n({combo_short})'
            else:
                value_str = f'{value:.1f}\n({combo_short})'
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   value_str, 
                   ha='left', va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Graph Edit Distance', fontsize=13, fontweight='bold')
        ax.set_title(f'{metric_label}\n(Lower is Better)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.invert_yaxis()  # Best (lowest) at top
    
    plt.suptitle('Graph Edit Distance Winners: Best (Lowest) GED by Strategy', 
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_all_metrics_charts(
    rankings_data: Dict,
    charts_dir: Path,
    split: str,
    suffix: str = ""
):
    """Create comprehensive charts showing all metrics across all models."""
    if not rankings_data or "rankings" not in rankings_data:
        return
    
    rankings = rankings_data["rankings"]
    if not rankings:
        return
    
    # Extract model names and metrics
    models = [entry["model"] for entry in rankings]
    
    # Extract metrics
    metrics_data = {
        "avg_macro_f1": [entry["avg_macro_f1"] for entry in rankings],
        "avg_macro_precision": [entry["avg_macro_precision"] for entry in rankings],
        "avg_macro_recall": [entry["avg_macro_recall"] for entry in rankings],
        "avg_fuzzy_f1": [entry["avg_fuzzy_f1"] for entry in rankings],
        "avg_exact_match_rate": [entry["avg_exact_match_rate"] for entry in rankings],
        "avg_omission_rate": [entry["avg_omission_rate"] for entry in rankings],
        "avg_hallucination_rate": [entry["avg_hallucination_rate"] for entry in rankings],
    }
    
    # Build filename suffix
    file_suffix = f"{suffix}_{split}" if suffix else split
    
    # Chart 1: Error Rates (Omission and Hallucination) - Lower is Better
    _create_error_rates_chart(models, metrics_data, charts_dir / f"all_metrics_error_rates_{file_suffix}.png")
    
    # Chart 2: Performance Metrics (F1, Precision, Recall) - Higher is Better
    _create_performance_metrics_chart(models, metrics_data, charts_dir / f"all_metrics_performance_{file_suffix}.png")
    
    # Chart 3: GED metrics (if available)
    if "model_details" in rankings_data:
        _create_ged_metrics_chart(rankings_data, charts_dir / f"all_metrics_ged_{file_suffix}.png")
    
    # Chart 4: Comprehensive view (all key metrics)
    _create_comprehensive_metrics_chart(models, metrics_data, charts_dir / f"all_metrics_comprehensive_{file_suffix}.png")


def _create_error_rates_chart(models: List[str], metrics_data: Dict, output_path: Path):
    """Create chart showing error rates (omission and hallucination) across all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    omission_rates = metrics_data["avg_omission_rate"]
    hallucination_rates = metrics_data["avg_hallucination_rate"]
    
    # Sort by omission rate (best to worst)
    sorted_indices = sorted(range(len(models)), key=lambda i: omission_rates[i])
    sorted_models = [models[i] for i in sorted_indices]
    sorted_omission = [omission_rates[i] for i in sorted_indices]
    sorted_hallucination = [hallucination_rates[i] for i in sorted_indices]
    
    # Omission Rate Chart
    colors_omission = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_models)))
    bars1 = ax1.barh(sorted_models, sorted_omission, color=colors_omission, alpha=0.8)
    ax1.set_xlabel('Omission Rate (Lower is Better)', fontsize=14, fontweight='bold')
    ax1.set_title('Omission Rate by Model\n(Missing Gold Relations)', fontsize=16, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.invert_yaxis()
    
    # Add value labels
    for bar, value in zip(bars1, sorted_omission):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', ha='left', va='center', fontsize=11, fontweight='bold')
    
    # Hallucination Rate Chart
    sorted_indices_hall = sorted(range(len(models)), key=lambda i: hallucination_rates[i])
    sorted_models_hall = [models[i] for i in sorted_indices_hall]
    sorted_hallucination_sorted = [hallucination_rates[i] for i in sorted_indices_hall]
    
    colors_hall = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_models_hall)))
    bars2 = ax2.barh(sorted_models_hall, sorted_hallucination_sorted, color=colors_hall, alpha=0.8)
    ax2.set_xlabel('Hallucination Rate (Lower is Better)', fontsize=14, fontweight='bold')
    ax2.set_title('Hallucination Rate by Model\n(False Positive Relations)', fontsize=16, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.invert_yaxis()
    
    # Add value labels
    for bar, value in zip(bars2, sorted_hallucination_sorted):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', ha='left', va='center', fontsize=11, fontweight='bold')
    
    plt.suptitle('Error Rates: Omission and Hallucination by Model', 
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_performance_metrics_chart(models: List[str], metrics_data: Dict, output_path: Path):
    """Create chart showing performance metrics (F1, Precision, Recall) across all models."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    metrics_to_plot = [
        ("avg_macro_f1", "Macro F1 Score", axes[0]),
        ("avg_macro_precision", "Macro Precision", axes[1]),
        ("avg_macro_recall", "Macro Recall", axes[2])
    ]
    
    for metric_key, metric_label, ax in metrics_to_plot:
        values = metrics_data[metric_key]
        
        # Sort by value (best to worst)
        sorted_indices = sorted(range(len(models)), key=lambda i: values[i], reverse=True)
        sorted_models = [models[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]
        
        # Use green gradient (darker green = better)
        colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(sorted_models)))
        bars = ax.barh(sorted_models, sorted_values, color=colors, alpha=0.8)
        
        ax.set_xlabel(f'{metric_label} (Higher is Better)', fontsize=14, fontweight='bold')
        ax.set_title(f'{metric_label} by Model', fontsize=16, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.invert_yaxis()
        
        # Add value labels
        for bar, value in zip(bars, sorted_values):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', ha='left', va='center', fontsize=11, fontweight='bold')
    
    plt.suptitle('Performance Metrics: F1, Precision, and Recall by Model', 
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_ged_metrics_chart(rankings_data: Dict, output_path: Path):
    """Create chart showing GED metrics across all models."""
    rankings = rankings_data["rankings"]
    model_details = rankings_data.get("model_details", {})
    
    if not model_details:
        return
    
    # Extract GED metrics from model details
    models = []
    avg_ged = []
    total_ged = []
    normalized_ged = []
    
    for entry in rankings:
        model = entry["model"]
        details = model_details.get(model, {})
        performances = details.get("performances", [])
        
        if performances:
            # Average GED across all performances
            ged_values = [p.get("avg_graph_edit_distance", 0) for p in performances if p.get("avg_graph_edit_distance", 0) > 0]
            total_ged_vals = [p.get("total_graph_edit_distance", 0) for p in performances if p.get("total_graph_edit_distance", 0) > 0]
            norm_ged_vals = [p.get("normalized_graph_edit_distance", 0) for p in performances if p.get("normalized_graph_edit_distance", 0) > 0]
            
            if ged_values:
                models.append(model)
                avg_ged.append(sum(ged_values) / len(ged_values))
                # Only add total/normalized GED if we have values
                if total_ged_vals:
                    total_ged.append(sum(total_ged_vals) / len(total_ged_vals))
                else:
                    total_ged.append(0.0)
                if norm_ged_vals:
                    normalized_ged.append(sum(norm_ged_vals) / len(norm_ged_vals))
                else:
                    normalized_ged.append(0.0)
    
    if not models:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Sort by normalized GED (best to worst)
    sorted_indices = sorted(range(len(models)), key=lambda i: normalized_ged[i] if normalized_ged[i] > 0 else avg_ged[i])
    sorted_models = [models[i] for i in sorted_indices]
    sorted_avg_ged = [avg_ged[i] for i in sorted_indices]
    sorted_total_ged = [total_ged[i] for i in sorted_indices]
    sorted_norm_ged = [normalized_ged[i] for i in sorted_indices]
    
    # Average GED
    colors1 = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_models)))
    bars1 = axes[0].barh(sorted_models, sorted_avg_ged, color=colors1, alpha=0.8)
    axes[0].set_xlabel('Average Graph Edit Distance (Lower is Better)', fontsize=14, fontweight='bold')
    axes[0].set_title('Average GED by Model', fontsize=16, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3, linestyle='--')
    axes[0].invert_yaxis()
    for bar, value in zip(bars1, sorted_avg_ged):
        width = bar.get_width()
        axes[0].text(width, bar.get_y() + bar.get_height()/2, 
                    f'{value:.1f}', ha='left', va='center', fontsize=11, fontweight='bold')
    
    # Total GED
    if any(t > 0 for t in sorted_total_ged):
        colors2 = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_models)))
        bars2 = axes[1].barh(sorted_models, sorted_total_ged, color=colors2, alpha=0.8)
        axes[1].set_xlabel('Total Graph Edit Distance (Lower is Better)', fontsize=14, fontweight='bold')
        axes[1].set_title('Total GED by Model', fontsize=16, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3, linestyle='--')
        axes[1].invert_yaxis()
        for bar, value in zip(bars2, sorted_total_ged):
            width = bar.get_width()
            axes[1].text(width, bar.get_y() + bar.get_height()/2, 
                        f'{value:.1f}', ha='left', va='center', fontsize=11, fontweight='bold')
    else:
        axes[1].axis('off')
    
    # Normalized GED
    if any(n > 0 for n in sorted_norm_ged):
        colors3 = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_models)))
        bars3 = axes[2].barh(sorted_models, sorted_norm_ged, color=colors3, alpha=0.8)
        axes[2].set_xlabel('Normalized GED (Lower is Better)', fontsize=14, fontweight='bold')
        axes[2].set_title('Normalized GED by Model\n(per gold relation)', fontsize=16, fontweight='bold')
        axes[2].grid(axis='x', alpha=0.3, linestyle='--')
        axes[2].invert_yaxis()
        for bar, value in zip(bars3, sorted_norm_ged):
            width = bar.get_width()
            axes[2].text(width, bar.get_y() + bar.get_height()/2, 
                        f'{value:.3f}', ha='left', va='center', fontsize=11, fontweight='bold')
    else:
        axes[2].axis('off')
    
    plt.suptitle('Graph Edit Distance Metrics by Model', 
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_comprehensive_metrics_chart(models: List[str], metrics_data: Dict, output_path: Path):
    """Create comprehensive chart showing all key metrics side by side."""
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    axes = axes.flatten()
    
    # Define all metrics to plot
    metrics_config = [
        ("avg_macro_f1", "Macro F1", True, axes[0]),
        ("avg_macro_precision", "Macro Precision", True, axes[1]),
        ("avg_macro_recall", "Macro Recall", True, axes[2]),
        ("avg_exact_match_rate", "Exact Match Rate", True, axes[3]),
        ("avg_omission_rate", "Omission Rate", False, axes[4]),
        ("avg_hallucination_rate", "Hallucination Rate", False, axes[5]),
    ]
    
    for metric_key, metric_label, higher_is_better, ax in metrics_config:
        values = metrics_data[metric_key]
        
        # Sort by value
        if higher_is_better:
            sorted_indices = sorted(range(len(models)), key=lambda i: values[i], reverse=True)
            colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(models)))
            xlim_max = 1.0
        else:
            sorted_indices = sorted(range(len(models)), key=lambda i: values[i])
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(models)))
            xlim_max = max(values) * 1.1 if values else 1.0
        
        sorted_models = [models[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]
        
        bars = ax.barh(sorted_models, sorted_values, color=colors, alpha=0.8)
        
        direction = "(Higher is Better)" if higher_is_better else "(Lower is Better)"
        ax.set_xlabel(f'{metric_label} {direction}', fontsize=12, fontweight='bold')
        ax.set_title(metric_label, fontsize=14, fontweight='bold')
        ax.set_xlim(0, xlim_max)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.invert_yaxis()
        
        # Add value labels
        for bar, value in zip(bars, sorted_values):
            width = bar.get_width()
            if higher_is_better:
                fmt = f'{value:.3f}'
            else:
                fmt = f'{value:.3f}'
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   fmt, ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.suptitle('Comprehensive Metrics Overview: All Models Performance', 
                 fontsize=20, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_strategy_ranking_chart(
    exact_comparison: Dict,
    output_path: Path
):
    """Create chart ranking all strategies by overall performance."""
    strategies = exact_comparison.get("strategies", [])
    strategy_summary = exact_comparison.get("strategy_summary", {})
    
    if not strategies or not strategy_summary:
        return
    
    # Collect metrics for each strategy
    strategy_metrics = []
    for strategy in strategies:
        summary = strategy_summary.get(strategy, {})
        strategy_metrics.append({
            "strategy": strategy,
            "macro_f1": summary.get("avg_macro_f1", 0.0),
            "macro_precision": summary.get("avg_macro_precision", 0.0),
            "macro_recall": summary.get("avg_macro_recall", 0.0),
            "micro_f1": summary.get("avg_micro_f1", 0.0),
            "avg_exact_match_rate": summary.get("avg_exact_match_rate", 0.0),
        })
    
    # Sort by macro F1 (primary metric)
    strategy_metrics.sort(key=lambda x: x["macro_f1"], reverse=True)
    
    strategies_ordered = [s["strategy"] for s in strategy_metrics]
    macro_f1_scores = [s["macro_f1"] for s in strategy_metrics]
    colors = [_get_strategy_color(s) for s in strategies_ordered]
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(strategies) * 0.6)))
    
    bars = ax.barh(strategies_ordered, macro_f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Average Macro F1 Score', fontsize=16, fontweight='bold')
    ax.set_title('Strategy Ranking by Performance\n(Average Macro F1 Across All Models)', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlim(0, max(macro_f1_scores) * 1.15 if macro_f1_scores else 1.0)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()  # Best at top
    
    # Add value labels and rank numbers
    for i, (bar, score) in enumerate(zip(bars, macro_f1_scores)):
        width = bar.get_width()
        rank = i + 1
        ax.text(width, bar.get_y() + bar.get_height()/2, 
               f'#{rank} {score:.4f}', ha='left', va='center', 
               fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_strategy_precision_recall_comparison(
    exact_comparison: Dict,
    output_path: Path
):
    """Create scatter/bar chart comparing precision and recall across strategies."""
    strategies = exact_comparison.get("strategies", [])
    strategy_summary = exact_comparison.get("strategy_summary", {})
    
    if not strategies or not strategy_summary:
        return
    
    precisions = []
    recalls = []
    colors_list = []
    
    for strategy in strategies:
        summary = strategy_summary.get(strategy, {})
        precisions.append(summary.get("avg_macro_precision", 0.0))
        recalls.append(summary.get("avg_macro_recall", 0.0))
        colors_list.append(_get_strategy_color(strategy))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Chart 1: Precision vs Recall scatter
    ax1.scatter(precisions, recalls, c=colors_list, s=200, alpha=0.7, edgecolors='black', linewidth=2)
    for i, strategy in enumerate(strategies):
        ax1.annotate(strategy, (precisions[i], recalls[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold')
    
    ax1.set_xlabel('Average Macro Precision', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average Macro Recall', fontsize=14, fontweight='bold')
    ax1.set_title('Strategy Comparison: Precision vs Recall', fontsize=16, fontweight='bold')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_xlim(0, max(max(precisions) * 1.1, 0.1) if precisions else 1.0)
    ax1.set_ylim(0, max(max(recalls) * 1.1, 0.1) if recalls else 1.0)
    
    # Chart 2: Side-by-side precision and recall bars
    x = np.arange(len(strategies))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, precisions, width, label='Precision', 
                   color='#3498db', alpha=0.8)
    bars2 = ax2.bar(x + width/2, recalls, width, label='Recall', 
                   color='#e74c3c', alpha=0.8)
    
    ax2.set_xlabel('Matching Strategy', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax2.set_title('Strategy Precision and Recall Comparison', fontsize=16, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=45, ha='right', fontsize=11)
    ax2.legend(fontsize=12)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_strategy_metrics_radar(
    exact_comparison: Dict,
    output_path: Path
):
    """Create radar chart comparing multiple metrics across strategies (top 6 strategies)."""
    strategies = exact_comparison.get("strategies", [])
    strategy_summary = exact_comparison.get("strategy_summary", {})
    
    if not strategies or not strategy_summary:
        return
    
    # Get top 6 strategies by macro F1
    strategy_f1 = [(s, strategy_summary.get(s, {}).get("avg_macro_f1", 0.0)) for s in strategies]
    strategy_f1.sort(key=lambda x: x[1], reverse=True)
    top_strategies = [s[0] for s in strategy_f1[:6]]  # Top 6
    
    if len(top_strategies) < 2:
        return  # Need at least 2 strategies
    
    # Define metrics to compare
    metrics = ['Macro F1', 'Macro Precision', 'Macro Recall', 'Micro F1', 'Exact Match Rate']
    metric_keys = ['avg_macro_f1', 'avg_macro_precision', 'avg_macro_recall', 
                   'avg_micro_f1', 'avg_exact_match_rate']
    
    # Set up radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    for strategy in top_strategies:
        summary = strategy_summary.get(strategy, {})
        values = [summary.get(key, 0.0) for key in metric_keys]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=strategy, 
               color=_get_strategy_color(strategy), alpha=0.7)
        ax.fill(angles, values, alpha=0.15, color=_get_strategy_color(strategy))
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    ax.set_title('Strategy Performance Radar Chart\n(Top 6 Strategies by Macro F1)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_strategy_correlation_heatmap(
    exact_comparison: Dict,
    output_path: Path
):
    """Create correlation heatmap showing how similar strategies perform."""
    strategies = exact_comparison.get("strategies", [])
    combinations = exact_comparison.get("combinations", {})

    if not strategies or not combinations:
        return

    # Need at least 2 strategies for correlation analysis
    if len(strategies) < 2:
        return
    
    # Build F1 score matrix: each row is a combination, each column is a strategy
    f1_matrix = []
    combo_names = []
    
    for combo_name in sorted(combinations.keys()):
        combo_data = combinations[combo_name]
        row = []
        for strategy in strategies:
            strategy_data = combo_data.get("strategies", {}).get(strategy, {})
            f1 = strategy_data.get("macro_f1", 0.0)
            row.append(f1)
        f1_matrix.append(row)
        combo_names.append(combo_name)
    
    if not f1_matrix:
        return
    
    # Compute correlation between strategies (columns)
    f1_array = np.array(f1_matrix)
    strategy_correlation = np.corrcoef(f1_array.T)
    
    fig, ax = plt.subplots(figsize=(max(10, len(strategies) * 1.2), max(8, len(strategies) * 1.0)))
    
    im = ax.imshow(strategy_correlation, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(strategies)))
    ax.set_yticks(np.arange(len(strategies)))
    ax.set_xticklabels(strategies, fontsize=11, rotation=45, ha='right')
    ax.set_yticklabels(strategies, fontsize=11)
    
    # Add text annotations
    for i in range(len(strategies)):
        for j in range(len(strategies)):
            text = ax.text(j, i, f'{strategy_correlation[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    ax.set_title("Strategy Performance Correlation\n(Correlation of F1 scores across all combinations)", 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# New chart functions for technique comparisons and overall rankings
# ============================================================================

def _create_technique_comparison_ranking(
    exact_comparison: Dict,
    output_path: Path,
    split: str
):
    """Create chart comparing model rankings across different techniques."""
    techniques_data = _group_comparisons_by_technique(exact_comparison)
    techniques = sorted(techniques_data.keys())
    
    if len(techniques) < 2:
        return  # Need at least 2 techniques to compare
    
    # Get all unique models across all techniques
    all_models = set()
    for technique, comp_data in techniques_data.items():
        combinations = comp_data.get("combinations", {})
        for combo_name in combinations.keys():
            model_part = _extract_model(combo_name)
            model = _reconstruct_model_name(model_part)
            all_models.add(model)
    
    all_models = sorted(all_models)
    
    # Build data: technique -> model -> best F1
    technique_model_f1 = {t: {} for t in techniques}
    for technique, comp_data in techniques_data.items():
        combinations = comp_data.get("combinations", {})
        strategies = comp_data.get("strategies", [])
        
        for model in all_models:
            best_f1 = 0.0
            for combo_name, combo_data in combinations.items():
                model_part = _extract_model(combo_name)
                combo_model = _reconstruct_model_name(model_part)
                if combo_model == model:
                    for strategy in strategies:
                        strategy_data = combo_data.get("strategies", {}).get(strategy, {})
                        f1 = strategy_data.get("macro_f1", 0.0)
                        best_f1 = max(best_f1, f1)
            technique_model_f1[technique][model] = best_f1
    
    # Get top models by average F1 across techniques
    model_avg_f1 = {}
    for model in all_models:
        f1s = [technique_model_f1[t].get(model, 0.0) for t in techniques]
        model_avg_f1[model] = np.mean(f1s) if f1s else 0.0
    
    top_models = sorted(model_avg_f1.items(), key=lambda x: x[1], reverse=True)[:15]
    models = [m[0] for m in top_models]
    
    # Build matrix for grouped bar chart
    x = np.arange(len(models))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(16, len(models) * 1.2), 8))
    
    technique_colors = plt.cm.Set3(np.linspace(0, 1, len(techniques)))
    for i, technique in enumerate(techniques):
        values = [technique_model_f1[technique].get(m, 0.0) for m in models]
        offset = (i - len(techniques)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=technique, 
                     color=technique_colors[i], alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Model', fontsize=16, fontweight='bold')
    ax.set_ylabel('Best Macro F1 Score', fontsize=16, fontweight='bold')
    ax.set_title('Technique Comparison: Model Performance Across Prompting Methods\n(Top 15 Models)', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=12)
    ax.legend(fontsize=13, title='Prompting Technique', title_fontsize=14)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_best_technique_per_model(
    exact_comparison: Dict,
    output_path: Path,
    split: str
):
    """Create chart showing which technique works best for each model."""
    techniques_data = _group_comparisons_by_technique(exact_comparison)
    techniques = sorted(techniques_data.keys())
    
    if len(techniques) < 2:
        return
    
    # Get all unique models
    all_models = set()
    for technique, comp_data in techniques_data.items():
        combinations = comp_data.get("combinations", {})
        for combo_name in combinations.keys():
            model_part = _extract_model(combo_name)
            model = _reconstruct_model_name(model_part)
            all_models.add(model)
    
    all_models = sorted(all_models)
    
    # Find best technique for each model using only exact matching
    model_best_technique = {}
    model_best_f1 = {}
    
    for model in all_models:
        best_technique = None
        best_f1 = -1.0
        
        for technique, comp_data in techniques_data.items():
            combinations = comp_data.get("combinations", {})
            
            for combo_name, combo_data in combinations.items():
                model_part = _extract_model(combo_name)
                combo_model = _reconstruct_model_name(model_part)
                if combo_model == model:
                    # Use only 'exact' strategy for exact comparison
                    strategies = combo_data.get("strategies", {})
                    if "exact" in strategies:
                        strategy_data = strategies["exact"]
                        f1 = strategy_data.get("macro_f1", 0.0)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_technique = technique
        
        if best_technique:
            model_best_technique[model] = best_technique
            model_best_f1[model] = best_f1
    
    if not model_best_technique:
        return
    
    # Sort by F1 score
    sorted_models = sorted(model_best_technique.items(), 
                          key=lambda x: model_best_f1[x[0]], reverse=True)
    
    models = [m[0] for m in sorted_models[:10]]
    best_techniques = [model_best_technique[m] for m in models]
    f1_scores = [model_best_f1[m] for m in models]
    
    # Color by technique
    technique_colors = {t: plt.cm.Set3(i/len(techniques)) 
                       for i, t in enumerate(techniques)}
    colors = [technique_colors.get(t, '#95a5a6') for t in best_techniques]
    
    fig, ax = plt.subplots(figsize=(14, max(8, len(models) * 0.4)))
    
    bars = ax.barh(models, f1_scores, color=colors, alpha=0.8)
    
    ax.set_xlabel('Best Macro F1 Score (With Relation Type Matching)', fontsize=16, fontweight='bold')
    ax.set_title('Best Prompting Technique per Model\n(Top 10 Models - With Relation Type Matching)', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlim(0, max(f1_scores) * 1.1 if f1_scores else 1.0)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()
    
    # Add value labels
    for bar, score, tech in zip(bars, f1_scores, best_techniques):
        width = bar.get_width()
        formatted_tech = _format_technique_name(tech)
        ax.text(width, bar.get_y() + bar.get_height()/2, 
               f'{score:.3f} ({formatted_tech})', ha='left', va='center', 
               fontsize=11, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=technique_colors[t], label=_format_technique_name(t)) 
                      for t in techniques]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=13)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_technique_metrics_comparison(
    exact_comparison: Dict,
    output_path: Path,
    split: str
):
    """Create chart comparing F1, precision, recall, grouped by technique."""
    combinations = exact_comparison.get("combinations", {})
    
    # Group data by technique (which will automatically group baseline/improved/full variants)
    technique_data = {}
    
    # Get all techniques
    all_techniques = set()
    
    for combo_key, combo_data in combinations.items():
        # Parse: "Baseline-CoT_model" or "Improved-IO_model" or "CoT_model"
        parts = combo_key.split('_', 1)
        if len(parts) < 2:
            continue
            
        technique_part = parts[0]
        
        # Determine technique and prompt level
        if technique_part.startswith('Baseline-'):
            technique = technique_part.replace('Baseline-', '')
            prompt_level = 'baseline'
        elif technique_part.startswith('Improved-'):
            technique = technique_part.replace('Improved-', '')
            prompt_level = 'improved'
        else:
            # Full version (no prefix)
            technique = technique_part
            prompt_level = 'full'
        
        all_techniques.add(technique)
        
        # Create combined technique name (e.g., "baseline-i/o", "improved-cot")
        tech_display = _format_technique_name(technique).lower()
        combined_name = f'{prompt_level}-{tech_display}'
        
        # Calculate metrics for this combination
        strategies = combo_data.get("strategies", {})
        f1s = []
        precisions = []
        recalls = []
        
        for strategy, strategy_data in strategies.items():
            f1 = strategy_data.get("macro_f1", 0.0)
            precision = strategy_data.get("macro_precision", 0.0)
            recall = strategy_data.get("macro_recall", 0.0)
            if f1 > 0:
                f1s.append(f1)
                precisions.append(precision)
                recalls.append(recall)
        
        if combined_name not in technique_data:
            technique_data[combined_name] = {
                "f1": [],
                "precision": [],
                "recall": []
            }
        
        technique_data[combined_name]["f1"].extend(f1s)
        technique_data[combined_name]["precision"].extend(precisions)
        technique_data[combined_name]["recall"].extend(recalls)
    
    if not all_techniques:
        return
    
    # Sort techniques: group by base technique, then by prompt level
    # Order: baseline-i/o, improved-i/o, full-i/o, baseline-cot, improved-cot, full-cot, etc.
    base_techniques = sorted(all_techniques)  # CoT, IO, RAG, ReAct
    prompt_levels = ['baseline', 'improved', 'full']
    
    # Create sorted list of combined technique names
    sorted_technique_names = []
    for base_tech in base_techniques:
        tech_display = _format_technique_name(base_tech).lower()
        for prompt_level in prompt_levels:
            combined_name = f'{prompt_level}-{tech_display}'
            if combined_name in technique_data:
                sorted_technique_names.append(combined_name)
    
    # Calculate averages for each technique × metric
    metrics_data = {}
    for tech_name in sorted_technique_names:
        if tech_name in technique_data:
            data = technique_data[tech_name]
            metrics_data[tech_name] = {
                "f1": np.mean(data["f1"]) if data["f1"] else 0.0,
                "precision": np.mean(data["precision"]) if data["precision"] else 0.0,
                "recall": np.mean(data["recall"]) if data["recall"] else 0.0
            }
        else:
            metrics_data[tech_name] = {
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0
            }
    
    # Create grouped bar chart
    # X-axis: techniques (baseline-i/o, improved-i/o, full-i/o, baseline-cot, ...)
    # Three bars per technique: F1, Precision, Recall
    x = np.arange(len(sorted_technique_names))
    width = 0.25  # Width for each metric bar
    
    fig, ax = plt.subplots(figsize=(max(18, len(sorted_technique_names) * 0.8), 8))
    
    # Metric colors
    metric_colors = {
        'f1': '#3498db',      # Blue
        'precision': '#2ecc71',  # Green
        'recall': '#e74c3c'   # Red
    }
    
    # Create bars: for each technique, show 3 metrics
    for j, metric in enumerate(['f1', 'precision', 'recall']):
        metric_offset = (j - 1) * width
        values = [metrics_data[tech_name][metric] for tech_name in sorted_technique_names]
        
        bars = ax.bar(x + metric_offset, values, width,
                     color=metric_colors[metric],
                     alpha=0.8,
                     edgecolor='white',
                     linewidth=0.5,
                     label=f'{metric.capitalize()}')
        
        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., val,
                       f'{val:.3f}', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=16, fontweight='bold')
    ax.set_title('Technique Comparison: Performance Metrics\n(F1, Precision, Recall)', 
                 fontsize=18, fontweight='bold', pad=30)
    
    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_technique_names, fontsize=10, rotation=45, ha='right')
    
    # Create custom legend showing metrics
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=metric_colors['f1'], label='Macro F1'),
        Patch(facecolor=metric_colors['precision'], label='Macro Precision'),
        Patch(facecolor=metric_colors['recall'], label='Macro Recall')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12, 
              title='Metric', title_fontsize=13, framealpha=0.9)
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Set y-axis limits
    all_values = []
    for tech_name in sorted_technique_names:
        all_values.extend([metrics_data[tech_name]['f1'], 
                          metrics_data[tech_name]['precision'], 
                          metrics_data[tech_name]['recall']])
    ax.set_ylim(0, max(max(all_values) * 1.2, 0.3) if all_values else 0.3)
    
    # Adjust margins to fit all labels
    plt.subplots_adjust(bottom=0.25, top=0.90)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_technique_model_heatmap(
    exact_comparison: Dict,
    output_path: Path,
    split: str
):
    """Create heatmap showing model performance by technique, grouped by prompt complexity."""
    combinations = exact_comparison.get("combinations", {})
    
    # Group by prompt level and technique
    prompt_levels = ['Baseline', 'Improved', 'Full']
    all_techniques = set()
    
    # Extract all techniques
    for combo_key in combinations.keys():
        parts = combo_key.split('_', 1)
        if len(parts) >= 2:
            technique_part = parts[0]
            if technique_part.startswith('Baseline-'):
                technique = technique_part.replace('Baseline-', '')
            elif technique_part.startswith('Improved-'):
                technique = technique_part.replace('Improved-', '')
            else:
                technique = technique_part
            all_techniques.add(technique)
    
    techniques = sorted(all_techniques)  # I/O, CoT, RAG, ReAct
    
    # Get all unique models
    all_models = set()
    for combo_key in combinations.keys():
        parts = combo_key.split('_', 1)
        if len(parts) >= 2:
            model_part = _extract_model(combo_key)
            model = _reconstruct_model_name(model_part)
            all_models.add(model)
    
    all_models = sorted(all_models)
    
    # Build F1 score matrix: rows = models, cols = [Baseline techniques, Improved techniques, Full techniques]
    # Column order: Baseline-I/O, Baseline-CoT, Baseline-RAG, Baseline-ReAct, Improved-I/O, ..., Full-ReAct
    f1_matrix = []
    column_labels = []
    
    for prompt_level in prompt_levels:
        for technique in techniques:
            column_labels.append(f'{prompt_level}-{_format_technique_name(technique)}')
    
    for model in all_models:
        row = []
        for prompt_level in prompt_levels:
            for technique in techniques:
                f1 = 0.0
                # Find matching combination
                for combo_key, combo_data in combinations.items():
                    parts = combo_key.split('_', 1)
                    if len(parts) < 2:
                        continue
                    
                    technique_part = parts[0]
                    model_part = _extract_model(combo_key)
                    combo_model = _reconstruct_model_name(model_part)
                    
                    # Check if this matches our prompt level and technique
                    if prompt_level == 'Baseline':
                        if technique_part.startswith('Baseline-') and technique_part.replace('Baseline-', '') == technique:
                            if combo_model == model:
                                strategies = combo_data.get("strategies", {})
                                # Baseline uses bertscore
                                if "bertscore" in strategies:
                                    f1 = strategies["bertscore"].get("macro_f1", 0.0)
                                break
                    elif prompt_level == 'Improved':
                        if technique_part.startswith('Improved-') and technique_part.replace('Improved-', '') == technique:
                            if combo_model == model:
                                strategies = combo_data.get("strategies", {})
                                if "exact" in strategies:
                                    f1 = strategies["exact"].get("macro_f1", 0.0)
                                break
                    else:  # Full
                        if not technique_part.startswith('Baseline-') and not technique_part.startswith('Improved-') and technique_part == technique:
                            if combo_model == model:
                                strategies = combo_data.get("strategies", {})
                                if "exact" in strategies:
                                    f1 = strategies["exact"].get("macro_f1", 0.0)
                                break
                row.append(f1)
        f1_matrix.append(row)
    
    if not f1_matrix:
        return
    
    fig, ax = plt.subplots(figsize=(max(16, len(column_labels) * 1.5), max(8, len(all_models) * 0.5)))
    
    max_val = max(max(row) for row in f1_matrix) if f1_matrix else 1.0
    im = ax.imshow(f1_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=max_val, 
                   extent=[-0.5, len(column_labels) - 0.5, len(all_models) - 0.5, -0.5])
    
    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_yticks(np.arange(len(all_models)))
    ax.set_yticklabels(all_models, fontsize=11)
    
    # Set x-axis labels - show technique names only, grouped by prompt level
    x_labels = []
    for prompt_level in prompt_levels:
        for technique in techniques:
            x_labels.append(_format_technique_name(technique))
    
    ax.set_xticklabels(x_labels, fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add vertical separators between groups
    n_techniques = len(techniques)
    for i in range(1, len(prompt_levels)):
        separator_pos = i * n_techniques - 0.5
        ax.axvline(x=separator_pos, color='black', linewidth=2, linestyle='-', alpha=0.5)
    
    # Set y-axis limits to show group labels
    ax.set_ylim(len(all_models) - 0.5, -1.5)
    
    # Add prompt level group labels above the heatmap
    for i, prompt_level in enumerate(prompt_levels):
        start_idx = i * n_techniques
        end_idx = start_idx + n_techniques - 1
        center = (start_idx + end_idx) / 2
        ax.text(center, -1.0, prompt_level, ha='center', va='top', 
               fontsize=14, fontweight='bold')
    
    # Add text annotations (bold the best result per model row)
    for i in range(len(all_models)):
        row_max = max(f1_matrix[i]) if f1_matrix[i] else 0  # Find best in this row
        for j in range(len(column_labels)):
            is_best = f1_matrix[i][j] == row_max and row_max > 0
            text = ax.text(j, i, f'{f1_matrix[i][j]:.3f}',
                          ha="center", va="center", color="black", fontsize=14,
                          fontweight='bold' if is_best else 'normal')
    
    ax.set_title("Model Performance by Prompting Technique\n(With Relation Type Matching)", 
                 fontsize=18, fontweight='bold', pad=30)
    ax.set_xlabel("Prompting Technique", fontsize=16, fontweight='bold')
    ax.set_ylabel("Model", fontsize=16, fontweight='bold')
    
    # Adjust margins
    plt.subplots_adjust(bottom=0.15, top=0.90)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Macro F1 Score', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_technique_ranking_chart(
    filtered_comparison: Dict,
    output_path: Path,
    technique: str,
    split: str
):
    """Create ranking chart from filtered comparison data for a specific technique."""
    combinations = filtered_comparison.get("combinations", {})
    strategies = filtered_comparison.get("strategies", [])
    
    if not combinations or not strategies:
        return
    
    # Build model rankings from filtered combinations
    model_scores = {}
    model_f1s = {}
    
    for combo_name, combo_data in combinations.items():
        model_part = _extract_model(combo_name)
        model = _reconstruct_model_name(model_part)
        
        # Find best F1 across all strategies for this model
        best_f1 = 0.0
        best_strategy_f1 = 0.0
        total_f1 = 0.0
        count = 0
        
        for strategy in strategies:
            strategy_data = combo_data.get("strategies", {}).get(strategy, {})
            f1 = strategy_data.get("macro_f1", 0.0)
            if f1 > 0:
                best_f1 = max(best_f1, f1)
                total_f1 += f1
                count += 1
        
        if count > 0:
            avg_f1 = total_f1 / count
        else:
            avg_f1 = best_f1
        
        # Use average F1 as score, or best F1 if no average
        score = avg_f1 if avg_f1 > 0 else best_f1
        
        if model not in model_scores or score > model_scores[model]:
            model_scores[model] = score
            model_f1s[model] = best_f1
    
    if not model_scores:
        return
    
    # Sort models by score
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    models = [m[0] for m in sorted_models[:15]]  # Top 15
    scores = [model_scores[m] for m in models]
    f1s = [model_f1s.get(m, 0.0) for m in models]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Chart 1: Overall Scores
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    bars1 = ax1.barh(models, scores, color=colors)
    ax1.set_xlabel('Best Average F1 Score', fontsize=16, fontweight='bold')
    ax1.set_title(f'Model Rankings: {technique} - With Relation Type Matching\nOverall Score Comparison', 
                  fontsize=18, fontweight='bold', pad=20)
    ax1.set_xlim(0, max(scores) * 1.1 if scores else 1.0)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.invert_yaxis()
    
    for bar, score in zip(bars1, scores):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', ha='left', va='center', fontsize=12)
    
    # Chart 2: F1 Scores
    ax2 = axes[1]
    bars2 = ax2.barh(models, f1s, color=colors)
    ax2.set_xlabel('Best Macro F1 Score', fontsize=16, fontweight='bold')
    ax2.set_title(f'Model Rankings: {technique} - With Relation Type Matching\nMacro F1 Score Comparison', 
                  fontsize=18, fontweight='bold', pad=20)
    ax2.set_xlim(0, max(f1s) * 1.1 if f1s else 1.0)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.invert_yaxis()
    
    for bar, score in zip(bars2, f1s):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', ha='left', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _filter_winners_by_technique(winners_data: Dict, technique: str) -> Dict:
    """Filter winners data to only include combinations for a specific technique."""
    filtered = {"by_strategy": {}, "by_metric": {}}
    
    by_strategy = winners_data.get("by_strategy", {})
    for strategy, strategy_winners in by_strategy.items():
        filtered_strategy = {}
        for metric_name, winner_info in strategy_winners.items():
            combo_name = winner_info.get("combination", "")
            if combo_name.startswith(technique + "_"):
                filtered_strategy[metric_name] = winner_info
        if filtered_strategy:
            filtered["by_strategy"][strategy] = filtered_strategy
    
    by_metric = winners_data.get("by_metric", {})
    for metric_name, metric_winners in by_metric.items():
        filtered_metric = {}
        for strategy, winner_info in metric_winners.items():
            combo_name = winner_info.get("combination", "")
            if combo_name.startswith(technique + "_"):
                filtered_metric[strategy] = winner_info
        if filtered_metric:
            filtered["by_metric"][metric_name] = filtered_metric
    
    return filtered


def _create_single_strategy_winners_chart(
    winners_data: Dict,
    output_path: Path,
    technique: Optional[str] = None
):
    """Create winners chart for a single strategy (used for per-technique winners)."""
    by_strategy = winners_data.get("by_strategy", {})
    
    if not by_strategy:
        return
    
    # Get the first (and likely only) strategy
    strategy = list(by_strategy.keys())[0]
    strategy_winners = by_strategy[strategy]
    
    if not strategy_winners:
        return
    
    # Define key metrics
    key_metrics = [
        ("macro_f1", "Macro F1"),
        ("macro_precision", "Macro Precision"),
        ("macro_recall", "Macro Recall"),
        ("micro_f1", "Micro F1"),
        ("micro_precision", "Micro Precision"),
        ("micro_recall", "Micro Recall"),
    ]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    metrics_labels = []
    combinations = []
    values = []
    
    for metric_name, metric_label in key_metrics:
        winner_info = strategy_winners.get(metric_name)
        if winner_info:
            metrics_labels.append(metric_label)
            combo_name = winner_info["combination"]
            # Extract model name from combo for cleaner display
            model_part = _extract_model(combo_name)
            model = _reconstruct_model_name(model_part)
            combinations.append(model)
            values.append(winner_info["value"])
    
    if not metrics_labels:
        plt.close()
        return
    
    y_pos = np.arange(len(metrics_labels))
    bars = ax.barh(y_pos, values, color=_get_strategy_color(strategy), alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics_labels, fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlabel('Score', fontsize=16, fontweight='bold')
    title = f'{strategy.upper()} Evaluation: Best Performers for Each Metric'
    if technique:
        title = f'{technique} - {title}'
    ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=1)
    
    # Add value labels and model names
    for idx, (bar, value, combo) in enumerate(zip(bars, values, combinations)):
        width = bar.get_width()
        ax.text(width + max(values) * 0.02, bar.get_y() + bar.get_height()/2, 
               f'{value:.4f}', 
               ha='left', va='center', fontsize=13, fontweight='bold')
        ax.text(width * 0.02, bar.get_y() + bar.get_height()/2, 
               combo, 
               ha='left', va='center', fontsize=11, color='white', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    ax.set_xlim(0, max(values) * 1.25)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# Wrapper functions for generating charts by category
# ============================================================================

def _generate_technique_charts(
    technique: str,
    exact_comparison: Dict,
    entity_only_comparison: Optional[Dict],
    exact_rankings: Optional[Dict],
    entity_only_rankings: Optional[Dict],
    charts_dir: Path,
    split: str,
    charts_generated: List[str],
    run_dir: Optional[Path] = None
):
    """Generate all charts for a specific prompting technique."""
    # Filter data by technique
    filtered_exact = _filter_by_technique(exact_comparison.get("combinations", {}), technique)
    filtered_entity = _filter_by_technique(entity_only_comparison.get("combinations", {}), technique) if entity_only_comparison else {}
    
    if not filtered_exact and not filtered_entity:
        return
    
    # Create technique directory structure
    technique_dir = charts_dir / "by_technique" / technique
    rankings_dir = technique_dir / "rankings"
    strategies_dir = technique_dir / "strategies"
    metrics_dir = technique_dir / "metrics"
    winners_dir = technique_dir / "winners"
    
    rankings_dir.mkdir(parents=True, exist_ok=True)
    strategies_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    winners_dir.mkdir(parents=True, exist_ok=True)
    
    # Rebuild comparison dicts with filtered combinations
    if filtered_exact:
        filtered_exact_comp = {
            "combinations": filtered_exact,
            "strategies": exact_comparison.get("strategies", []),
            "strategy_summary": {}  # Will be recalculated if needed
        }
        
        # Generate strategy charts for this technique
        _create_best_strategy_per_model(
            filtered_exact_comp,
            strategies_dir / f"best_strategy_per_model_{technique}_exact_{split}.png",
            technique=technique
        )
        charts_generated.append(f"by_technique/{technique}/strategies/best_strategy_per_model_{technique}_exact_{split}.png")
        
        _create_strategy_model_heatmap(
            filtered_exact_comp,
            strategies_dir / f"strategy_model_heatmap_{technique}_exact_{split}.png"
        )
        charts_generated.append(f"by_technique/{technique}/strategies/strategy_model_heatmap_{technique}_exact_{split}.png")
        
        _create_strategy_performance_by_model(
            filtered_exact_comp,
            strategies_dir / f"strategy_performance_by_model_{technique}_exact_{split}.png"
        )
        charts_generated.append(f"by_technique/{technique}/strategies/strategy_performance_by_model_{technique}_exact_{split}.png")
        
        _create_strategy_ranking_chart(
            filtered_exact_comp,
            strategies_dir / f"strategy_ranking_{technique}_exact_{split}.png"
        )
        charts_generated.append(f"by_technique/{technique}/strategies/strategy_ranking_{technique}_exact_{split}.png")
        
        _create_strategy_precision_recall_comparison(
            filtered_exact_comp,
            strategies_dir / f"strategy_precision_recall_{technique}_exact_{split}.png"
        )
        charts_generated.append(f"by_technique/{technique}/strategies/strategy_precision_recall_{technique}_exact_{split}.png")
        
        # Generate metrics charts
        _create_aggregated_counts_chart(
            filtered_exact_comp,
            metrics_dir / f"aggregated_counts_{technique}_exact_{split}.png"
        )
        charts_generated.append(f"by_technique/{technique}/metrics/aggregated_counts_{technique}_exact_{split}.png")
        
        _create_overall_rates_chart(
            filtered_exact_comp,
            metrics_dir / f"overall_rates_{technique}_exact_{split}.png"
        )
        charts_generated.append(f"by_technique/{technique}/metrics/overall_rates_{technique}_exact_{split}.png")
    
    if filtered_entity:
        filtered_entity_comp = {
            "combinations": filtered_entity,
            "strategies": entity_only_comparison.get("strategies", []),
            "strategy_summary": {}
        }
        
        _create_best_strategy_per_model(
            filtered_entity_comp,
            strategies_dir / f"best_strategy_per_model_{technique}_entity_only_{split}.png",
            technique=technique
        )
        charts_generated.append(f"by_technique/{technique}/strategies/best_strategy_per_model_{technique}_entity_only_{split}.png")
        
        _create_strategy_model_heatmap(
            filtered_entity_comp,
            strategies_dir / f"strategy_model_heatmap_{technique}_entity_only_{split}.png"
        )
        charts_generated.append(f"by_technique/{technique}/strategies/strategy_model_heatmap_{technique}_entity_only_{split}.png")
        
        _create_strategy_performance_by_model(
            filtered_entity_comp,
            strategies_dir / f"strategy_performance_by_model_{technique}_entity_only_{split}.png"
        )
        charts_generated.append(f"by_technique/{technique}/strategies/strategy_performance_by_model_{technique}_entity_only_{split}.png")
        
        # Generate metrics charts
        _create_aggregated_counts_chart(
            filtered_entity_comp,
            metrics_dir / f"aggregated_counts_{technique}_entity_only_{split}.png"
        )
        charts_generated.append(f"by_technique/{technique}/metrics/aggregated_counts_{technique}_entity_only_{split}.png")
        
        _create_overall_rates_chart(
            filtered_entity_comp,
            metrics_dir / f"overall_rates_{technique}_entity_only_{split}.png"
        )
        charts_generated.append(f"by_technique/{technique}/metrics/overall_rates_{technique}_entity_only_{split}.png")
    
    # Generate rankings charts from filtered comparison data
    if filtered_exact:
        _create_technique_ranking_chart(
            filtered_exact_comp,
            rankings_dir / f"model_rankings_{technique}_exact_{split}.png",
            technique=technique,
            split=split
        )
        charts_generated.append(f"by_technique/{technique}/rankings/model_rankings_{technique}_exact_{split}.png")
    
    if filtered_entity:
        _create_technique_ranking_chart(
            filtered_entity_comp,
            rankings_dir / f"model_rankings_{technique}_entity_only_{split}.png",
            technique=technique,
            split=split
        )
        charts_generated.append(f"by_technique/{technique}/rankings/model_rankings_{technique}_entity_only_{split}.png")
    
    # Generate winners charts for this technique
    # Load winners data if available
    if run_dir is None:
        run_dir = charts_dir.parent
    winners_file = run_dir / f"winners_{split}.json"
    if winners_file.exists():
        try:
            with open(winners_file, 'r', encoding='utf-8') as f:
                all_winners_data = json.load(f)
            
            # Filter winners by technique
            filtered_winners = _filter_winners_by_technique(all_winners_data, technique)
            
            if filtered_winners.get("by_strategy"):
                # Generate overview winners chart
                _create_winners_chart(
                    filtered_winners,
                    winners_dir / f"winners_{technique}_{split}.png",
                    technique=technique
                )
                charts_generated.append(f"by_technique/{technique}/winners/winners_{technique}_{split}.png")
                
                # Generate individual strategy winner charts
                for strategy in filtered_winners.get("by_strategy", {}).keys():
                    strategy_winners = {"by_strategy": {strategy: filtered_winners["by_strategy"][strategy]}}
                    if strategy_winners["by_strategy"][strategy]:
                        output_file = winners_dir / f"winners_{technique}_{strategy}_{split}.png"
                        # Create a simple winners chart for this strategy
                        _create_single_strategy_winners_chart(
                            strategy_winners,
                            output_file,
                            technique=technique
                        )
                        charts_generated.append(f"by_technique/{technique}/winners/winners_{technique}_{strategy}_{split}.png")
                
                # Generate GED winners chart if available
                _create_ged_winners_chart(
                    filtered_winners,
                    winners_dir / f"ged_winners_{technique}_{split}.png"
                )
                charts_generated.append(f"by_technique/{technique}/winners/ged_winners_{technique}_{split}.png")
        except Exception as e:
            # Silently skip if winners file can't be loaded
            pass


def _generate_technique_comparison_charts(
    exact_comparison: Dict,
    entity_only_comparison: Optional[Dict],
    charts_dir: Path,
    split: str,
    charts_generated: List[str]
):
    """Generate charts comparing different prompting techniques."""
    comparisons_dir = charts_dir / "technique_comparisons"
    rankings_dir = comparisons_dir / "rankings"
    strategies_dir = comparisons_dir / "strategies"
    models_dir = comparisons_dir / "models"
    overall_dir = comparisons_dir / "overall"
    
    rankings_dir.mkdir(parents=True, exist_ok=True)
    strategies_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    overall_dir.mkdir(parents=True, exist_ok=True)
    
    if exact_comparison:
        # Technique comparison charts
        _create_technique_comparison_ranking(
            exact_comparison,
            rankings_dir / f"technique_comparison_model_rankings_exact_{split}.png",
            split
        )
        charts_generated.append(f"technique_comparisons/rankings/technique_comparison_model_rankings_exact_{split}.png")
        
        _create_best_technique_per_model(
            exact_comparison,
            models_dir / f"best_technique_per_model_exact_{split}.png",
            split
        )
        charts_generated.append(f"technique_comparisons/models/best_technique_per_model_exact_{split}.png")
        
        _create_technique_metrics_comparison(
            exact_comparison,
            overall_dir / f"technique_metrics_comparison_exact_{split}.png",
            split
        )
        charts_generated.append(f"technique_comparisons/overall/technique_metrics_comparison_exact_{split}.png")
        
        _create_technique_model_heatmap(
            exact_comparison,
            models_dir / f"technique_model_heatmap_exact_{split}.png",
            split
        )
        charts_generated.append(f"technique_comparisons/models/technique_model_heatmap_exact_{split}.png")
    
    if entity_only_comparison:
        _create_technique_model_heatmap(
            entity_only_comparison,
            models_dir / f"technique_model_heatmap_entity_only_{split}.png",
            split
        )
        charts_generated.append(f"technique_comparisons/models/technique_model_heatmap_entity_only_{split}.png")
        
        _create_technique_metrics_comparison(
            entity_only_comparison,
            overall_dir / f"technique_metrics_comparison_entity_only_{split}.png",
            split
        )
        charts_generated.append(f"technique_comparisons/overall/technique_metrics_comparison_entity_only_{split}.png")
        
        _create_technique_comparison_ranking(
            entity_only_comparison,
            rankings_dir / f"technique_comparison_model_rankings_entity_only_{split}.png",
            split
        )
        charts_generated.append(f"technique_comparisons/rankings/technique_comparison_model_rankings_entity_only_{split}.png")
        
        _create_best_technique_per_model(
            entity_only_comparison,
            models_dir / f"best_technique_per_model_entity_only_{split}.png",
            split
        )
        charts_generated.append(f"technique_comparisons/models/best_technique_per_model_entity_only_{split}.png")


def _generate_overall_charts(
    exact_comparison: Dict,
    entity_only_comparison: Optional[Dict],
    exact_rankings: Optional[Dict],
    entity_only_rankings: Optional[Dict],
    charts_dir: Path,
    split: str,
    charts_generated: List[str]
):
    """Generate overall aggregated charts across all techniques."""
    overall_dir = charts_dir / "overall"
    rankings_dir = overall_dir / "rankings"
    strategies_dir = overall_dir / "strategies"
    metrics_dir = overall_dir / "metrics"
    winners_dir = overall_dir / "winners"
    
    rankings_dir.mkdir(parents=True, exist_ok=True)
    strategies_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    winners_dir.mkdir(parents=True, exist_ok=True)
    
    # Overall charts use aggregated data (already aggregated in exact_comparison)
    if exact_comparison:
        _create_best_strategy_per_model(
            exact_comparison,
            strategies_dir / f"overall_best_strategy_per_model_exact_{split}.png",
            technique=None  # Overall, so no specific technique
        )
        charts_generated.append(f"overall/strategies/overall_best_strategy_per_model_exact_{split}.png")
        
        _create_strategy_model_heatmap(
            exact_comparison,
            strategies_dir / f"overall_strategy_model_heatmap_exact_{split}.png"
        )
        charts_generated.append(f"overall/strategies/overall_strategy_model_heatmap_exact_{split}.png")
    
    if entity_only_comparison:
        _create_best_strategy_per_model(
            entity_only_comparison,
            strategies_dir / f"overall_best_strategy_per_model_entity_only_{split}.png",
            technique=None
        )
        charts_generated.append(f"overall/strategies/overall_best_strategy_per_model_entity_only_{split}.png")
        
        _create_strategy_model_heatmap(
            entity_only_comparison,
            strategies_dir / f"overall_strategy_model_heatmap_entity_only_{split}.png"
        )
        charts_generated.append(f"overall/strategies/overall_strategy_model_heatmap_entity_only_{split}.png")
    
    # Generate overall rankings if available
    if exact_rankings:
        _create_model_ranking_chart(
            exact_rankings,
            entity_only_rankings,
            rankings_dir / f"overall_model_rankings_exact_{split}.png",
            technique=None
        )
        charts_generated.append(f"overall/rankings/overall_model_rankings_exact_{split}.png")
    
    if entity_only_rankings:
        _create_entity_only_ranking_chart(
            entity_only_rankings,
            rankings_dir / f"overall_model_rankings_entity_only_{split}.png",
            technique=None
        )
        charts_generated.append(f"overall/rankings/overall_model_rankings_entity_only_{split}.png")
    
    # Generate overall metrics charts
    if exact_rankings:
        _create_all_metrics_charts(
            exact_rankings,
            metrics_dir,
            split,
            suffix=f"overall_exact"
        )
        charts_generated.extend([
            f"overall/metrics/all_metrics_error_rates_overall_exact_{split}.png",
            f"overall/metrics/all_metrics_performance_overall_exact_{split}.png",
            f"overall/metrics/all_metrics_ged_overall_exact_{split}.png",
            f"overall/metrics/all_metrics_comprehensive_overall_exact_{split}.png"
        ])
    
    if entity_only_rankings:
        _create_all_metrics_charts(
            entity_only_rankings,
            metrics_dir,
            split,
            suffix=f"overall_entity_only"
        )
        charts_generated.extend([
            f"overall/metrics/all_metrics_error_rates_overall_entity_only_{split}.png",
            f"overall/metrics/all_metrics_performance_overall_entity_only_{split}.png",
            f"overall/metrics/all_metrics_ged_overall_entity_only_{split}.png",
            f"overall/metrics/all_metrics_comprehensive_overall_entity_only_{split}.png"
        ])

