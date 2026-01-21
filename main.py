"""Main pipeline orchestration.

This script runs the complete relation extraction pipeline:
1. Loads test data (documents and gold relations)
2. Builds global entity map
3. Runs all prompting techniques on all documents
4. Parses LLM responses
5. Evaluates predictions against gold standard
6. Aggregates and compares results across techniques
"""

import json
import logging
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from config import Config
from utils.logging import setup_logger, get_log_file_path

from pipeline.data import DatasetLoader, GlobalEntityMap
from pipeline.llm_prompter import (
    IOPrompter,
    ChainOfThoughtPrompter,
    RAGPrompter,
    ReActPrompter,
    BaselinePrompter,
)
from pipeline.parsing import ResponseParser
from pipeline.evaluation import Evaluator, DocumentExporter, get_available_matchers
from pipeline.evaluation.document_logger import DocumentLogger
from pipeline.aggregation import ResultAggregator, TechniqueComparator, ModelRanker
from pipeline.cache import LLMResponseCache

from pipeline.types import Document, GoldRelations, ParsedRelations, EvaluationResult, AggregateResults


def main(
    split: str = "train",
    models: Optional[Dict[str, List[str]]] = None,
    techniques: Optional[List[str]] = None,
    max_documents: Optional[int] = None,
    max_workers: Optional[int] = None,
    parallel_combinations: bool = False,
    matching_strategies: Optional[List[str]] = None,
):
    """
    Run the complete relation extraction pipeline with multiple models and matching strategies.
    
    Args:
        split: Data split to use ("dev", "test", or "train")
        models: Optional dict mapping technique names to list of model keys
                e.g., {"IO": ["gpt-4o-mini", "gpt-4o"], "CoT": ["gpt-4o-mini"]}
                If a technique is not specified, uses default models
        techniques: Optional list of techniques to run (defaults to all)
                   e.g., ["IO", "CoT", "RAG", "ReAct"]
        max_documents: Optional limit on number of documents to process (for testing)
        max_workers: Maximum number of parallel workers for document processing.
                     Defaults to min(8, number of documents) if None.
        parallel_combinations: If True, process different technique-model combinations
                              in parallel. If False, process sequentially (default).
        matching_strategies: List of matching strategies to test (defaults to all)
                            e.g., ["exact", "fuzzy", "text", "bertscore"]
    """
    # ========== Configuration ==========
    Config.validate()
    
    clean_text_path = Config.CLEAN_TEXT_PATH
    gold_relations_path = Config.GOLD_RELATIONS_PATH
    
    # Create run-specific directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Config.OUTPUT_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== Save Prompt Templates (for tracking changes) ==========
    prompts_source = Path(__file__).parent / "pipeline" / "prompts.py"
    prompts_hash = None
    if prompts_source.exists():
        # Copy prompts file to run directory
        prompts_dest = run_dir / "prompts.py"
        shutil.copy2(prompts_source, prompts_dest)
        
        # Calculate hash for quick comparison
        with open(prompts_source, "rb") as f:
            prompts_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        
        # Save prompts metadata
        prompts_metadata = {
            "source_file": str(prompts_source.relative_to(Path(__file__).parent.parent)),
            "run_timestamp": timestamp,
            "prompts_hash": prompts_hash,
            "file_size_bytes": prompts_source.stat().st_size,
            "last_modified": datetime.fromtimestamp(prompts_source.stat().st_mtime).isoformat(),
            "note": "This file tracks the prompt templates used for this run. Compare prompts.py and prompts_hash across runs to identify prompt changes that may affect results."
        }
        with open(run_dir / "prompts_metadata.json", "w") as f:
            json.dump(prompts_metadata, f, indent=2)
    
    # ========== Setup Logging ==========
    log_file = run_dir / f"pipeline_{split}_{timestamp}.log" if Config.LOG_TO_FILE else None
    log_level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
    
    logger = setup_logger(
        name="pipeline",
        log_file=log_file,
        level=log_level,
        console=Config.LOG_TO_CONSOLE
    )
    
    if matching_strategies is None:
        matching_strategies = ["exact"]
    
    logger.info(f"Matching strategies to test: {matching_strategies}")
    logger.info(f"Available matchers: {get_available_matchers()}")
    
    # Create subdirectories for each matching strategy (exact matching with types)
    strategy_dirs = {}
    for strategy in matching_strategies:
        strategy_dir = run_dir / f"matching_{strategy}"
        strategy_dir.mkdir(exist_ok=True)
        summaries_dir = strategy_dir / "summaries"
        summaries_dir.mkdir(exist_ok=True)
        strategy_dirs[strategy] = {
            "main": strategy_dir,
            "summaries": summaries_dir
        }
    
    # Create subdirectories for entity-only matching (without types)
    entity_only_strategy_dirs = {}
    for strategy in matching_strategies:
        entity_only_dir = run_dir / f"entity_only_matching_{strategy}"
        entity_only_dir.mkdir(exist_ok=True)
        entity_only_summaries_dir = entity_only_dir / "summaries"
        entity_only_summaries_dir.mkdir(exist_ok=True)
        entity_only_strategy_dirs[strategy] = {
            "main": entity_only_dir,
            "summaries": entity_only_summaries_dir
        }
    
    # Create summaries subdirectory in main run_dir for overall comparison
    summaries_dir = run_dir / "summaries"
    summaries_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Starting Relation Extraction Pipeline")
    logger.info("=" * 80)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Split: {split}")
    logger.info(f"Max documents: {max_documents if max_documents else 'All'}")
    if log_file:
        logger.info(f"Log file: {log_file}")
    if prompts_hash:
        logger.info(f"Prompts hash: {prompts_hash} (prompts.py saved to run directory)")
    
    # Default to all techniques if not specified
    if techniques is None:
        techniques = ["IO", "CoT", "RAG", "ReAct"]
    
    # Default models (can be overridden)
    # If models is None or empty, use default models for each technique
    if models is None:
        models = {}
    
    # Set default models for techniques that don't have models specified
    # Full list of models to test (from figures analysis)
    default_models = [
        "anthropic/claude-sonnet-4.5",      # Anthropic - strong reasoning
        "openai/gpt-4.1",                   # OpenAI - smartest non-reasoning model
        "openai/gpt-4o",                    # OpenAI - GPT-4o
        "openai/gpt-4o-mini",               # OpenAI - GPT-4o Mini
        "openai/gpt-5-mini",                # OpenAI - faster, cost-efficient
        "openai/gpt-5-nano",                # OpenAI - smallest/fastest
        "google/gemini-2.0-flash-001",      # Google - Gemini 2.0 Flash
        "deepseek/deepseek-chat-v3.1",      # DeepSeek - Chat v3.1
        "meta-llama/llama-3.1-70b-instruct",# Meta - Llama 3.1 70B
        "mistralai/mistral-nemo",           # Mistral - Nemo
    ]
    for technique in techniques:
        if technique not in models or not models[technique]:
            models[technique] = default_models.copy()
            logger.info(f"Using default models {default_models} for {technique}")
    
    # Log model configuration
    logger.info("\nModel Configuration:")
    for technique in techniques:
        model_list = models.get(technique, [])
        logger.info(f"  {technique}: {model_list}")
    
    # ========== Step 1: Load Data ==========
    logger.info("=" * 80)
    logger.info("Step 1: Loading data...")
    logger.info("=" * 80)
    loader = DatasetLoader(clean_text_path, gold_relations_path, logger=logger)
    documents, gold_relations = loader.load(split)
    # logger.info(f"Loaded {len(documents)} documents")
    
    # Log gold relations file names
    logger.info("Gold relations files loaded:")
#   for gold in gold_relations:
#       file_name = Path(gold.file_path).name if gold.file_path else "unknown"
#       logger.info(f"  - {file_name} (doc_id: {gold.doc_id}, {len(gold.relations)} relations)")
    
    # Limit documents if max_documents is specified
    if max_documents and max_documents > 0:
        original_count = len(documents)
        documents = documents[:max_documents]
        gold_relations = gold_relations[:max_documents]
        logger.info(f"Limited to {len(documents)} documents (from {original_count})")

    # Set up parallelism (now that documents are loaded)
    if max_workers is None:
        max_workers = min(8, len(documents)) if documents else 4
    logger.info(f"\nParallelism Configuration:")
    logger.info(f"  Max workers for document processing: {max_workers}")
    logger.info(f"  Parallel combinations: {parallel_combinations}")

    # ========== Step 2: Build Global Entity Map ==========
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Building global entity map...")
    logger.info("=" * 80)
    entity_map = GlobalEntityMap()
    entity_map.build_from_gold_relations(gold_relations)
    logger.info(f"Entity map contains {len(entity_map)} entities")
    
    # ========== Step 3: Process with Each Matching Strategy ==========
    # Store results for all strategies: strategy -> {combo_name -> results}
    all_strategy_results: Dict[str, Dict[str, List[EvaluationResult]]] = {}
    all_strategy_aggregated: Dict[str, Dict[str, AggregateResults]] = {}
    
    # Process documents once and reuse for all strategies
    # First, we need to process all documents with all techniques/models
    # Then evaluate with each matching strategy
    
    # Initialize parser (shared across all strategies)
    parser = ResponseParser(entity_map=entity_map, logger=logger)
    
    # Initialize LLM response cache
    llm_cache = LLMResponseCache(
        cache_dir=Config.LLM_CACHE_DIR,
        enabled=Config.USE_LLM_CACHE or Config.SAVE_LLM_CACHE,
        logger=logger
    )
    
    # Log cache stats
    cache_stats = llm_cache.get_stats()
    if cache_stats.get("enabled"):
        logger.info(
            f"[LLMCache] Cache stats: {cache_stats['total_files']} files, "
            f"{cache_stats['total_size_mb']} MB"
        )
        if Config.USE_LLM_CACHE:
            logger.info("[LLMCache] Using cached responses when available")
        if Config.SAVE_LLM_CACHE:
            logger.info("[LLMCache] Saving new responses to cache")
    
    # Process all combinations once, then evaluate with each strategy
    # This avoids reprocessing documents multiple times
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Processing documents with all techniques/models...")
    logger.info("=" * 80)
    
    # Store predictions: combo_name -> (predictions, prompts, raw_responses)
    all_predictions: Dict[str, tuple] = {}
    
    # Process documents (this is done once, shared across strategies)
    _process_all_combinations(
        documents=documents,
        gold_relations=gold_relations,
        techniques=techniques,
        models=models,
        entity_map=entity_map,
        parser=parser,
        all_predictions=all_predictions,
        max_workers=max_workers,
        parallel_combinations=parallel_combinations,
        llm_cache=llm_cache,
        logger=logger
    )
    
    # Now evaluate with each matching strategy (both with and without type matching)
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Evaluating with each matching strategy...")
    logger.info("=" * 80)
    logger.info("Note: Each strategy will be evaluated twice:")
    logger.info("  1. With type matching (exact matches only)")
    logger.info("  2. Without type matching (entity-only, partial matches count as correct)")
    
    # Store results for both evaluation types
    all_strategy_results = {}  # For exact matching (with types)
    all_strategy_aggregated = {}  # For exact matching (with types)
    all_entity_only_strategy_results = {}  # For entity-only matching (without types)
    all_entity_only_strategy_aggregated = {}  # For entity-only matching (without types)
    
    for strategy in matching_strategies:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Evaluating with matching strategy: {strategy}")
        logger.info(f"{'=' * 80}")
        
        # ========== EVALUATION 1: WITH TYPE MATCHING (EXACT) ==========
        logger.info(f"\n--- Evaluation 1: With Type Matching (Exact) ---")
        strategy_dir = strategy_dirs[strategy]["main"]
        strategy_summaries_dir = strategy_dirs[strategy]["summaries"]
        
        # Create evaluator with strategy
        matcher_config = {}
        if strategy == "bertscore":
            matcher_config["use_openai_embeddings"] = Config.USE_OPENAI_EMBEDDINGS_FOR_BERTSCORE
            matcher_config["similarity_threshold"] = Config.BERTSCORE_THRESHOLD
        elif strategy == "text":
            matcher_config["similarity_threshold"] = Config.BERTSCORE_THRESHOLD
            matcher_config["fuzzy_threshold"] = Config.TEXT_MATCHING_FUZZY_THRESHOLD
        elif strategy == "jaccard":
            matcher_config["similarity_threshold"] = Config.JACCARD_THRESHOLD
        elif strategy == "token":
            matcher_config["similarity_threshold"] = Config.TOKEN_THRESHOLD
            matcher_config["use_token_set"] = Config.TOKEN_USE_TOKEN_SET
        elif strategy == "levenshtein":
            matcher_config["similarity_threshold"] = Config.LEVENSHTEIN_THRESHOLD
        elif strategy == "jaro_winkler":
            matcher_config["similarity_threshold"] = Config.JARO_WINKLER_THRESHOLD
        elif strategy == "sbert":
            matcher_config["similarity_threshold"] = Config.SBERT_THRESHOLD
            matcher_config["model_name"] = Config.SBERT_MODEL
        
        evaluator_exact = Evaluator(
            strategy=strategy,
            entity_map=entity_map, 
            match_type=True,
            matcher_config=matcher_config,
            logger=logger
        )
        document_logger_exact = DocumentLogger(
            output_dir=strategy_dir,
            entity_map=entity_map,
            logger=logger
        )
        aggregator = ResultAggregator(logger=logger)
        comparator = TechniqueComparator()
        
        all_results: Dict[str, List[EvaluationResult]] = {}
        aggregated_results: Dict[str, AggregateResults] = {}
        
        for combo_name, (predictions, prompts, raw_responses) in all_predictions.items():
            logger.info(f"Evaluating {combo_name} with {strategy} matching (with types)...")
            eval_results = evaluator_exact.evaluate(predictions, gold_relations)
            all_results[combo_name] = eval_results
            
            # Log per-document results
            parts = combo_name.split("_", 1)
            technique = parts[0] if len(parts) > 0 else "unknown"
            model = parts[1] if len(parts) > 1 else "unknown"
            
            for eval_result, pred, gold_rel, prompt, raw_response in zip(eval_results, predictions, gold_relations, prompts, raw_responses):
                stats = {
                    "precision": eval_result.precision,
                    "recall": eval_result.recall,
                    "f1_score": eval_result.f1_score,
                    "fuzzy_precision": eval_result.fuzzy_precision,
                    "fuzzy_recall": eval_result.fuzzy_recall,
                    "fuzzy_f1": eval_result.fuzzy_f1,
                    "exact_match_rate": eval_result.exact_match_rate,
                    "omission_rate": eval_result.omission_rate,
                    "hallucination_rate": eval_result.hallucination_rate,
                    "redundancy_rate": eval_result.redundancy_rate,
                    "graph_edit_distance": eval_result.graph_edit_distance,
                    "bertscore": eval_result.bertscore,
                    "num_true_positives": len(eval_result.true_positives),
                    "num_false_positives": len(eval_result.false_positives),
                    "num_false_negatives": len(eval_result.false_negatives),
                    "num_partial_matches": len(eval_result.partial_matches),
                    "num_predicted_relations": len(pred.relations),
                    "num_gold_relations": len(gold_rel.relations),
                    "matching_strategy": strategy,
                    "match_type": "with_types",
                    "technique": technique,
                    "model": model,
                }
                
                document_logger_exact.log_document_results(
                    technique_name=combo_name,
                    doc_id=eval_result.doc_id,
                    eval_result=eval_result,
                    predicted_relations=pred.relations,
                    gold_relations_obj=gold_rel,
                    stats=stats,
                    prompt=prompt,
                    raw_response=raw_response
                )
            
            aggregated = aggregator.aggregate(eval_results, combo_name)
            aggregated_results[combo_name] = aggregated
            
            summary_path = strategy_summaries_dir / f"{combo_name}_summary.json"
            summary = {
                "technique": technique,
                "model": model,
                "combination": combo_name,
                "matching_strategy": strategy,
                "match_type": "with_types",
                "split": split,
                "num_documents": len(eval_results),
                "timestamp": timestamp,
                "exact_match_metrics": {
                    "macro_precision": aggregated.macro_precision,
                    "macro_recall": aggregated.macro_recall,
                    "macro_f1": aggregated.macro_f1,
                    "micro_precision": aggregated.micro_precision,
                    "micro_recall": aggregated.micro_recall,
                    "micro_f1": aggregated.micro_f1,
                    "avg_exact_match_rate": aggregated.avg_exact_match_rate,
                    "avg_omission_rate": aggregated.avg_omission_rate,
                    "avg_hallucination_rate": aggregated.avg_hallucination_rate,
                    "avg_redundancy_rate": aggregated.avg_redundancy_rate,
                    "avg_graph_edit_distance": aggregated.avg_graph_edit_distance,
                    "total_graph_edit_distance": aggregated.total_graph_edit_distance,
                    "normalized_graph_edit_distance": aggregated.normalized_graph_edit_distance,
                },
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
                "fuzzy_match_metrics": {
                    "total_partial_matches": aggregated.total_partial_matches,
                    "avg_partial_matches": aggregated.avg_partial_matches,
                    "fuzzy_macro_precision": aggregated.fuzzy_macro_precision,
                    "fuzzy_macro_recall": aggregated.fuzzy_macro_recall,
                    "fuzzy_macro_f1": aggregated.fuzzy_macro_f1,
                    "fuzzy_micro_precision": aggregated.fuzzy_micro_precision,
                    "fuzzy_micro_recall": aggregated.fuzzy_micro_recall,
                    "fuzzy_micro_f1": aggregated.fuzzy_micro_f1,
                }
            }
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
        
        # Generate per-document detailed JSON exports
        logger.info(f"Generating per-document detailed JSON exports for strategy: {strategy}")
        json_export_dir = strategy_dir / "document_details"
        json_export_dir.mkdir(exist_ok=True)
        
        for combo_name, (predictions, _, _) in all_predictions.items():
            eval_results = all_results[combo_name]
            # Export detailed JSON for each document
            for eval_result, pred, gold_rel in zip(eval_results, predictions, gold_relations):
                output_path = json_export_dir / f"{combo_name}_{eval_result.doc_id}_{strategy}.json"
                DocumentExporter.export_document_details(
                    eval_result.doc_id,
                    strategy,
                    pred,
                    gold_rel,
                    eval_result,
                    output_path
                )
        
        report_path = strategy_dir / f"comparison_{split}.json"
        comparator.save_report(aggregated_results, str(report_path))
        logger.info(f"Saved {strategy} comparison report (with types) to: {report_path}")
        logger.info(f"Saved per-document detailed JSON exports to: {json_export_dir}")
        
        all_strategy_results[strategy] = all_results
        all_strategy_aggregated[strategy] = aggregated_results
        
        # ========== EVALUATION 2: WITHOUT TYPE MATCHING (ENTITY-ONLY) ==========
        logger.info(f"\n--- Evaluation 2: Without Type Matching (Entity-Only) ---")
        entity_only_dir = entity_only_strategy_dirs[strategy]["main"]
        entity_only_summaries_dir = entity_only_strategy_dirs[strategy]["summaries"]
        
        evaluator_entity_only = Evaluator(
            strategy=strategy,
            entity_map=entity_map, 
            match_type=False,
            matcher_config=matcher_config,
            logger=logger
        )
        document_logger_entity_only = DocumentLogger(
            output_dir=entity_only_dir,
            entity_map=entity_map,
            logger=logger
        )
        aggregator_entity_only = ResultAggregator(logger=logger)
        comparator_entity_only = TechniqueComparator()
        
        all_entity_only_results: Dict[str, List[EvaluationResult]] = {}
        aggregated_entity_only_results: Dict[str, AggregateResults] = {}
        
        for combo_name, (predictions, prompts, raw_responses) in all_predictions.items():
            logger.info(f"Evaluating {combo_name} with {strategy} matching (entity-only, no types)...")
            eval_results = evaluator_entity_only.evaluate(predictions, gold_relations)
            all_entity_only_results[combo_name] = eval_results
            
            # Log per-document results
            parts = combo_name.split("_", 1)
            technique = parts[0] if len(parts) > 0 else "unknown"
            model = parts[1] if len(parts) > 1 else "unknown"
            
            for eval_result, pred, gold_rel, prompt, raw_response in zip(eval_results, predictions, gold_relations, prompts, raw_responses):
                stats = {
                    "precision": eval_result.precision,
                    "recall": eval_result.recall,
                    "f1_score": eval_result.f1_score,
                    "fuzzy_precision": eval_result.fuzzy_precision,
                    "fuzzy_recall": eval_result.fuzzy_recall,
                    "fuzzy_f1": eval_result.fuzzy_f1,
                    "exact_match_rate": eval_result.exact_match_rate,
                    "omission_rate": eval_result.omission_rate,
                    "hallucination_rate": eval_result.hallucination_rate,
                    "redundancy_rate": eval_result.redundancy_rate,
                    "graph_edit_distance": eval_result.graph_edit_distance,
                    "bertscore": eval_result.bertscore,
                    "num_true_positives": len(eval_result.true_positives),
                    "num_false_positives": len(eval_result.false_positives),
                    "num_false_negatives": len(eval_result.false_negatives),
                    "num_partial_matches": len(eval_result.partial_matches),
                    "num_predicted_relations": len(pred.relations),
                    "num_gold_relations": len(gold_rel.relations),
                    "matching_strategy": strategy,
                    "match_type": "entity_only",
                    "technique": technique,
                    "model": model,
                }
                
                document_logger_entity_only.log_document_results(
                    technique_name=combo_name,
                    doc_id=eval_result.doc_id,
                    eval_result=eval_result,
                    predicted_relations=pred.relations,
                    gold_relations_obj=gold_rel,
                    stats=stats,
                    prompt=prompt,
                    raw_response=raw_response
                )
            
            aggregated = aggregator_entity_only.aggregate(eval_results, combo_name)
            aggregated_entity_only_results[combo_name] = aggregated
            
            summary_path = entity_only_summaries_dir / f"{combo_name}_summary.json"
            summary = {
                "technique": technique,
                "model": model,
                "combination": combo_name,
                "matching_strategy": strategy,
                "match_type": "entity_only",
                "split": split,
                "num_documents": len(eval_results),
                "timestamp": timestamp,
                "exact_match_metrics": {
                    "macro_precision": aggregated.macro_precision,
                    "macro_recall": aggregated.macro_recall,
                    "macro_f1": aggregated.macro_f1,
                    "micro_precision": aggregated.micro_precision,
                    "micro_recall": aggregated.micro_recall,
                    "micro_f1": aggregated.micro_f1,
                    "avg_exact_match_rate": aggregated.avg_exact_match_rate,
                    "avg_omission_rate": aggregated.avg_omission_rate,
                    "avg_hallucination_rate": aggregated.avg_hallucination_rate,
                    "avg_redundancy_rate": aggregated.avg_redundancy_rate,
                    "avg_graph_edit_distance": aggregated.avg_graph_edit_distance,
                    "total_graph_edit_distance": aggregated.total_graph_edit_distance,
                    "normalized_graph_edit_distance": aggregated.normalized_graph_edit_distance,
                },
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
                "fuzzy_match_metrics": {
                    "total_partial_matches": aggregated.total_partial_matches,
                    "avg_partial_matches": aggregated.avg_partial_matches,
                    "fuzzy_macro_precision": aggregated.fuzzy_macro_precision,
                    "fuzzy_macro_recall": aggregated.fuzzy_macro_recall,
                    "fuzzy_macro_f1": aggregated.fuzzy_macro_f1,
                    "fuzzy_micro_precision": aggregated.fuzzy_micro_precision,
                    "fuzzy_micro_recall": aggregated.fuzzy_micro_recall,
                    "fuzzy_micro_f1": aggregated.fuzzy_micro_f1,
                }
            }
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
        
        # Generate per-document detailed JSON exports for entity-only matching
        logger.info(f"Generating per-document detailed JSON exports for strategy: {strategy} (entity-only)")
        json_export_dir_entity_only = entity_only_dir / "document_details"
        json_export_dir_entity_only.mkdir(exist_ok=True)
        
        for combo_name, (predictions, _, _) in all_predictions.items():
            eval_results = all_entity_only_results[combo_name]
            # Export detailed JSON for each document
            for eval_result, pred, gold_rel in zip(eval_results, predictions, gold_relations):
                output_path = json_export_dir_entity_only / f"{combo_name}_{eval_result.doc_id}_{strategy}_entity_only.json"
                DocumentExporter.export_document_details(
                    eval_result.doc_id,
                    f"{strategy}_entity_only",
                    pred,
                    gold_rel,
                    eval_result,
                    output_path
                )
        
        report_path = entity_only_dir / f"comparison_{split}.json"
        comparator_entity_only.save_report(aggregated_entity_only_results, str(report_path))
        logger.info(f"Saved {strategy} comparison report (entity-only) to: {report_path}")
        logger.info(f"Saved per-document detailed JSON exports (entity-only) to: {json_export_dir_entity_only}")
        
        all_entity_only_strategy_results[strategy] = all_entity_only_results
        all_entity_only_strategy_aggregated[strategy] = aggregated_entity_only_results
    
    # ========== Step 5: Compare All Matching Strategies ==========
    from pipeline.evaluation import compare_matching_strategies
    
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Comparing matching strategies (with type matching)...")
    logger.info("=" * 80)
    compare_matching_strategies(
        all_strategy_aggregated,
        matching_strategies,
        run_dir,
        split,
        logger
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("Step 5b: Comparing matching strategies (entity-only matching)...")
    logger.info("=" * 80)
    # Generate entity-only comparison and save to separate file
    compare_matching_strategies(
        all_entity_only_strategy_aggregated,
        matching_strategies,
        run_dir,
        split,
        logger
    )
    # Copy to entity_only filename (original file will be regenerated for exact matching)
    temp_path = run_dir / f"strategy_comparison_{split}.json"
    entity_only_path = run_dir / f"strategy_comparison_{split}_entity_only.json"
    if temp_path.exists():
        shutil.copy2(temp_path, entity_only_path)
        logger.info(f"Saved entity-only comparison to: {entity_only_path}")
    
    # Regenerate regular comparison (with types)
    compare_matching_strategies(
        all_strategy_aggregated,
        matching_strategies,
        run_dir,
        split,
        logger
    )
    
    # ========== Step 6: Rank Models Across All Techniques ==========
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: Ranking models across all techniques and strategies...")
    logger.info("=" * 80)
    
    model_ranker = ModelRanker(logger=logger)
    
    # Rank models with type matching (exact)
    logger.info("\n--- Rankings: With Type Matching (Exact) ---")
    ranking_report = model_ranker.rank_models(
        all_strategy_aggregated,
        matching_strategies,
        run_dir,
        split
    )
    
    # Rank models without type matching (entity-only)
    logger.info("\n--- Rankings: Entity-Only Matching (Partial Matches = Correct) ---")
    entity_only_ranking_report = model_ranker.rank_models(
        all_entity_only_strategy_aggregated,
        matching_strategies,
        run_dir,
        split
    )
    # Copy to entity_only filename (original will be regenerated)
    temp_ranking = run_dir / f"model_rankings_{split}.json"
    entity_only_ranking = run_dir / f"model_rankings_{split}_entity_only.json"
    if temp_ranking.exists():
        shutil.copy2(temp_ranking, entity_only_ranking)
        logger.info(f"Saved entity-only rankings to: {entity_only_ranking}")
    
    # Regenerate regular rankings (they were overwritten)
    ranking_report = model_ranker.rank_models(
        all_strategy_aggregated,
        matching_strategies,
        run_dir,
        split
    )
    
    # ========== Step 7: Generate Evaluation Charts ==========
    logger.info("\n" + "=" * 80)
    logger.info("Step 7: Generating evaluation charts...")
    logger.info("=" * 80)
    
    try:
        from pipeline.visualization import generate_all_charts
        
        # Load comparison data for charts
        exact_comparison_file = run_dir / f"strategy_comparison_{split}.json"
        entity_only_comparison_file = run_dir / f"strategy_comparison_{split}_entity_only.json"
        
        exact_comparison_data = None
        entity_only_comparison_data = None
        
        if exact_comparison_file.exists():
            with open(exact_comparison_file, 'r', encoding='utf-8') as f:
                exact_comparison_data = json.load(f)
        
        if entity_only_comparison_file.exists():
            with open(entity_only_comparison_file, 'r', encoding='utf-8') as f:
                entity_only_comparison_data = json.load(f)
        
        # Load ranking data for charts
        exact_ranking_file = run_dir / f"model_rankings_{split}.json"
        entity_only_ranking_file = run_dir / f"model_rankings_{split}_entity_only.json"
        
        exact_ranking_data = None
        entity_only_ranking_data = None
        
        if exact_ranking_file.exists():
            with open(exact_ranking_file, 'r', encoding='utf-8') as f:
                exact_ranking_data = json.load(f)
        
        if entity_only_ranking_file.exists():
            with open(entity_only_ranking_file, 'r', encoding='utf-8') as f:
                entity_only_ranking_data = json.load(f)
        
        # Generate charts
        generate_all_charts(
            run_dir=run_dir,
            split=split,
            exact_rankings=exact_ranking_data,
            entity_only_rankings=entity_only_ranking_data,
            exact_comparison=exact_comparison_data,
            entity_only_comparison=entity_only_comparison_data,
            logger=logger
        )
    except ImportError:
        logger.warning("Chart generation not available (matplotlib may not be installed)")
    except Exception as e:
        logger.warning(f"Error generating charts: {e}", exc_info=True)
    
    logger.info("\n" + "=" * 80)
    logger.info("Pipeline completed!")
    logger.info("=" * 80)
    logger.info(f"Processed {len(documents)} documents")
    logger.info(f"Tested {len(matching_strategies)} matching strategies")
    logger.info(f"Results saved to: {run_dir}")
    logger.info("\nEvaluation Results:")
    for strategy in matching_strategies:
        logger.info(f"  With types ({strategy}): {strategy_dirs[strategy]['main']}")
        logger.info(f"  Entity-only ({strategy}): {entity_only_strategy_dirs[strategy]['main']}")
    charts_dir = run_dir / "charts"
    if charts_dir.exists() and list(charts_dir.glob("*.png")):
        logger.info(f"\nEvaluation Charts:")
        logger.info(f"  Charts directory: {charts_dir}")
        for chart_file in sorted(charts_dir.glob("*.png")):
            logger.info(f"    - {chart_file.name}")
    
    # ========== Final Cache Statistics ==========
    if llm_cache.enabled:
        final_cache_stats = llm_cache.get_stats()
        logger.info("\n" + "=" * 80)
        logger.info("LLM Cache Statistics:")
        logger.info("=" * 80)
        logger.info(f"Total cached responses: {final_cache_stats['total_files']}")
        logger.info(f"Total cache size: {final_cache_stats['total_size_mb']} MB")
        logger.info(f"Cache directory: {final_cache_stats['cache_dir']}")
        logger.info("\nNote: Cached responses can be reused in future runs to avoid re-querying LLMs.")
        logger.info("      Set USE_LLM_CACHE=True to use cached responses, SAVE_LLM_CACHE=True to save new ones.")
    
    logger.info("\n" + "=" * 80)
    logger.info("Pipeline completed!")
    logger.info("=" * 80)
    if log_file:
        logger.info(f"Log file: {log_file}")


def _process_all_combinations(
    documents: List[Document],
    gold_relations: List[GoldRelations],
    techniques: List[str],
    models: Dict[str, List[str]],
    entity_map: GlobalEntityMap,
    parser: ResponseParser,
    all_predictions: Dict[str, tuple],
    max_workers: int,
    parallel_combinations: bool,
    llm_cache: LLMResponseCache,
    logger: logging.Logger
):
    """Process all technique-model combinations and store predictions."""
    # Helper function to create prompter
    def create_prompter(technique: str, model: str):
        """Create a prompter for the given technique and model."""
        # Determine prompt_mode based on technique prefix
        # Supports: "Baseline-IO", "Improved-IO", "IO" (full)
        if technique.startswith("Baseline-"):
            prompt_mode = "baseline"
            base_technique = technique.replace("Baseline-", "")
        elif technique.startswith("Improved-"):
            prompt_mode = "improved"
            base_technique = technique.replace("Improved-", "")
        else:
            prompt_mode = "full"
            base_technique = technique
        
        # For baseline mode, don't include relation types in the minimal prompt
        include_relation_types = Config.INCLUDE_RELATION_TYPES and prompt_mode != "baseline"
        
        if base_technique == "IO":
            return IOPrompter(
                entity_map=entity_map,
                use_exact_spans=True,
                include_relation_types=include_relation_types,
                model=model,
                prompt_mode=prompt_mode,
                logger=logger
            )
        elif base_technique == "CoT":
            return ChainOfThoughtPrompter(
                entity_map=entity_map,
                use_exact_spans=True,
                include_relation_types=include_relation_types,
                model=model,
                prompt_mode=prompt_mode,
                logger=logger
            )
        elif base_technique == "RAG":
            return RAGPrompter(
                entity_map=entity_map,
                use_exact_spans=True,
                include_relation_types=include_relation_types,
                model=model,
                prompt_mode=prompt_mode,
                logger=logger
            )
        elif base_technique == "ReAct":
            return ReActPrompter(
                entity_map=entity_map,
                use_exact_spans=True,
                include_relation_types=include_relation_types,
                model=model,
                prompt_mode=prompt_mode,
                logger=logger
            )
        elif technique == "Baseline":  # Legacy support
            return BaselinePrompter(
                entity_map=entity_map,
                use_exact_spans=True,
                include_relation_types=False,
                model=model,
                logger=logger
            )
        else:
            raise ValueError(f"Unknown technique: {technique}")
    
    # Build list of all combinations
    combinations = []
    for technique in techniques:
        model_list = models.get(technique, [])
        for model in model_list:
            combinations.append((technique, model))
    
    total_combinations = len(combinations)
    logger.info(f"Processing {total_combinations} technique-model combinations...")
    
    # Process all combinations in parallel if enabled
    if parallel_combinations:
        # Thread-safe lock for updating shared dictionary
        predictions_lock = Lock()
        log_lock = Lock()
        
        def process_combination(technique: str, model: str) -> tuple[str, tuple]:
            """Process a single combination and return (combo_name, (predictions, prompts, raw_responses))."""
            combo_name = f"{technique}_{model.replace('/', '_').replace('-', '_')}"
            try:
                with log_lock:
                    logger.info(f"Starting: {combo_name}")
                
                # Create prompter
                prompter = create_prompter(technique, model)
                
                # Process documents (still in parallel within each combination if multiple docs)
                predictions, prompts, raw_responses = _process_documents_parallel(
                    documents=documents,
                    gold_relations=gold_relations,
                    prompter=prompter,
                    parser=parser,
                    max_workers=max_workers if len(documents) > 1 else 1,  # Use 1 worker if single doc to avoid overhead
                    llm_cache=llm_cache,
                    technique=technique,
                    model=model,
                    logger=logger
                )
                
                # Sort predictions by doc_id to match gold_relations order
                doc_data = list(zip(predictions, prompts, raw_responses))
                doc_data.sort(key=lambda x: x[0].doc_id or "")
                if doc_data:
                    predictions, prompts, raw_responses = zip(*doc_data)
                    predictions = list(predictions)
                    prompts = list(prompts)
                    raw_responses = list(raw_responses)
                else:
                    predictions, prompts, raw_responses = [], [], []
                
                with log_lock:
                    logger.info(f"Completed: {combo_name} ({len(predictions)} documents)")
                
                return combo_name, (predictions, prompts, raw_responses)
                
            except Exception as e:
                with log_lock:
                    logger.error(f"Error processing {combo_name}: {e}", exc_info=True)
                # Return empty predictions on error
                empty_predictions = [ParsedRelations(doc_id=doc.doc_id) for doc in documents]
                empty_prompts = [""] * len(documents)
                empty_responses = [""] * len(documents)
                return combo_name, (empty_predictions, empty_prompts, empty_responses)
        
        # Process all combinations in parallel
        # Use a reasonable number of workers: either max_workers or number of combinations, whichever is smaller
        combination_workers = min(max_workers * 2, total_combinations)  # Allow more parallelism for combinations
        logger.info(f"Processing {total_combinations} combinations in parallel with {combination_workers} workers")
        
        with ThreadPoolExecutor(max_workers=combination_workers) as executor:
            # Submit all combination tasks
            future_to_combo = {
                executor.submit(process_combination, technique, model): (technique, model)
                for technique, model in combinations
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_combo):
                technique, model = future_to_combo[future]
                try:
                    combo_name, result = future.result()
                    with predictions_lock:
                        all_predictions[combo_name] = result
                    completed += 1
                    with log_lock:
                        logger.info(f"Progress: {completed}/{total_combinations} combinations completed")
                except Exception as e:
                    combo_name = f"{technique}_{model.replace('/', '_').replace('-', '_')}"
                    with log_lock:
                        logger.error(f"Error getting result for {combo_name}: {e}", exc_info=True)
                    # Add empty predictions on error
                    empty_predictions = [ParsedRelations(doc_id=doc.doc_id) for doc in documents]
                    empty_prompts = [""] * len(documents)
                    empty_responses = [""] * len(documents)
                    with predictions_lock:
                        all_predictions[combo_name] = (empty_predictions, empty_prompts, empty_responses)
        
        logger.info(f"Completed processing all {total_combinations} combinations")
    
    else:
        # Process each combination sequentially (original behavior)
        for technique, model in combinations:
            combo_name = f"{technique}_{model.replace('/', '_').replace('-', '_')}"
            logger.info(f"Processing: {combo_name}")
            
            # Create prompter
            prompter = create_prompter(technique, model)
            
            # Process documents
            predictions, prompts, raw_responses = _process_documents_parallel(
                documents=documents,
                gold_relations=gold_relations,
                prompter=prompter,
                parser=parser,
                max_workers=max_workers,
                llm_cache=llm_cache,
                technique=technique,
                model=model,
                logger=logger
            )
            
            # Sort predictions by doc_id to match gold_relations order
            doc_data = list(zip(predictions, prompts, raw_responses))
            doc_data.sort(key=lambda x: x[0].doc_id or "")
            if doc_data:
                predictions, prompts, raw_responses = zip(*doc_data)
                predictions = list(predictions)
                prompts = list(prompts)
                raw_responses = list(raw_responses)
            else:
                predictions, prompts, raw_responses = [], [], []
            
            # Store predictions, prompts, and raw responses
            all_predictions[combo_name] = (predictions, prompts, raw_responses)
            logger.info(f"Completed: {combo_name} ({len(predictions)} documents)")
    


def _process_documents_parallel(
    documents: List[Document],
    gold_relations: List[GoldRelations],
    prompter,
    parser,
    max_workers: int,
    llm_cache: LLMResponseCache,
    technique: str,
    model: str,
    logger: logging.Logger
) -> tuple[List[ParsedRelations], List[str], List[str]]:
    """
    Process documents in parallel.
    
    Args:
        documents: List of documents to process
        gold_relations: List of GoldRelations (one per document, same order as documents)
        prompter: Prompter instance to use
        parser: Parser instance to use
        max_workers: Maximum number of parallel workers
        llm_cache: LLM response cache
        technique: Prompting technique name
        model: Model name
        logger: Logger instance
        
    Returns:
        Tuple of (List of ParsedRelations, List of prompts, List of raw responses)
    """
    if len(documents) == 0:
        return []
    
    # Create mapping from doc_id to gold_relations for quick lookup
    gold_relations_by_doc_id = {gr.doc_id: gr for gr in gold_relations}
    
    # Thread-safe lock for logging
    log_lock = Lock()
    
    def process_single_document(doc: Document) -> tuple[str, ParsedRelations, str, str]:
        """Process a single document and return (doc_id, parsed_relations, prompt, raw_response)."""
        try:
            with log_lock:
                logger.info(f"Processing document: {doc.doc_id}")
            
            # Get document entity IDs from gold relations (if available)
            document_entity_ids = None
            gold_rel = gold_relations_by_doc_id.get(doc.doc_id)
            if gold_rel and gold_rel.entities:
                document_entity_ids = {entity.id for entity in gold_rel.entities}
                with log_lock:
                    logger.debug(f"[Parser] Document {doc.doc_id}: Limiting entity resolution to {len(document_entity_ids)} entities in document")
            
            # Check cache first (before building prompt to avoid unnecessary work)
            # This is especially important for RAG to avoid retrieval when cached
            raw_response = ""
            cached_data = None
            prompt = ""
            
            if Config.USE_LLM_CACHE:
                # Try to find cached response by document text first
                # This works for all techniques and avoids building prompt/retrieval if cached
                cached_data = llm_cache.get_by_document(technique, model, doc.doc_id, doc.text)
                if cached_data:
                    raw_response = cached_data.get("response", "")
                    prompt = cached_data.get("prompt", "")
                    with log_lock:
                        logger.info(
                            f"[Cache HIT] Using cached response for {doc.doc_id} "
                            f"(skipped prompt building{' and retrieval' if technique == 'RAG' else ''})"
                        )
            
            # Build prompt only if not cached
            if not raw_response:
                prompt = prompter._build_prompt(doc.text, doc_id=doc.doc_id)
                
                # Check cache again with full prompt (in case document-based lookup failed)
                # This handles cases where prompt structure changed but document is same
                if Config.USE_LLM_CACHE and not cached_data:
                    cached_data = llm_cache.get(technique, model, doc.doc_id, prompt)
                    if cached_data:
                        raw_response = cached_data.get("response", "")
                        with log_lock:
                            logger.info(f"[Cache HIT] Using cached response for {doc.doc_id} (by prompt hash)")
            
            # If not in cache, get from LLM
            if not raw_response:
                # Get LLM response (this is the full raw response)
                raw_response = prompter.get_response(doc.text, doc_id=doc.doc_id)
                
                # Save to cache
                if Config.SAVE_LLM_CACHE and raw_response:
                    llm_cache.save(
                        technique=technique,
                        model=model,
                        doc_id=doc.doc_id,
                        prompt=prompt,
                        response=raw_response,
                        document_text=doc.text,  # Pass document text for better cache lookup
                        metadata={
                            "technique": technique,
                            "model": model,
                            "doc_id": doc.doc_id
                        }
                    )
            
            # Parse response with document entity IDs for better resolution
            parsed = parser.parse(
                raw_response,
                doc_id=doc.doc_id,
                source_text=doc.text,
                document_entity_ids=document_entity_ids
            )
            parsed.doc_id = doc.doc_id  # Ensure doc_id is set
            
            with log_lock:
                logger.info(
                    f"Document {doc.doc_id}: Parsed {len(parsed.relations)} relations, "
                    f"{len(parsed.parsing_errors)} parsing errors, "
                    f"{len(parsed.entity_resolution_errors)} resolution errors"
                )
            
            return doc.doc_id, parsed, prompt, raw_response
            
        except Exception as e:
            with log_lock:
                # Error details already logged by prompter, just log summary
                error_msg = str(e)
                if "429" in error_msg or "Rate limit" in error_msg:
                    logger.error(f"Error processing document {doc.doc_id}: Rate limit exceeded (details logged above)")
                else:
                    logger.error(f"Error processing document {doc.doc_id}: {error_msg[:200]}")
            # Return empty parsed relations on error
            parsed = ParsedRelations(doc_id=doc.doc_id)
            # Try to get prompt even on error
            try:
                prompt = prompter._build_prompt(doc.text, doc_id=doc.doc_id)
            except:
                prompt = ""
            raw_response = ""  # Empty response on error
            return doc.doc_id, parsed, prompt, raw_response
    
    # Process documents in parallel
    predictions_dict: Dict[str, ParsedRelations] = {}
    prompts_dict: Dict[str, str] = {}
    responses_dict: Dict[str, str] = {}  # Store raw LLM responses
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_doc = {
            executor.submit(process_single_document, doc): doc
            for doc in documents
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_doc):
            doc = future_to_doc[future]
            try:
                doc_id, parsed, prompt, raw_response = future.result()
                predictions_dict[doc_id] = parsed
                prompts_dict[doc_id] = prompt
                responses_dict[doc_id] = raw_response
                completed += 1
                with log_lock:
                    logger.info(f"Progress: {completed}/{len(documents)} documents completed")
            except Exception as e:
                with log_lock:
                    logger.error(f"Error getting result for document {doc.doc_id}: {e}", exc_info=True)
                # Add empty parsed relations on error
                predictions_dict[doc.doc_id] = ParsedRelations(doc_id=doc.doc_id)
                prompts_dict[doc.doc_id] = ""
                responses_dict[doc.doc_id] = ""
    
    # Return predictions, prompts, and raw responses in the same order as documents
    predictions = [predictions_dict.get(doc.doc_id, ParsedRelations(doc_id=doc.doc_id)) 
                   for doc in documents]
    prompts = [prompts_dict.get(doc.doc_id, "") for doc in documents]
    raw_responses = [responses_dict.get(doc.doc_id, "") for doc in documents]
    
    return predictions, prompts, raw_responses


def _analyze_model_performance(
    aggregated_results: Dict[str, AggregateResults],
    techniques: List[str],
    models_config: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Analyze model performance across techniques.
    
    Returns:
        Dictionary with model comparison analysis
    """
    # Group results by technique and model
    by_technique: Dict[str, Dict[str, AggregateResults]] = {}
    by_model: Dict[str, Dict[str, AggregateResults]] = {}
    
    for combo_name, results in aggregated_results.items():
        # Parse combo_name (format: "technique_model")
        parts = combo_name.split("_", 1)
        if len(parts) == 2:
            technique = parts[0]
            model = parts[1].replace("_", "-").replace("/", "/")
            
            if technique not in by_technique:
                by_technique[technique] = {}
            by_technique[technique][model] = results
            
            if model not in by_model:
                by_model[model] = {}
            by_model[model][technique] = results
    
    # Find best model for each technique
    best_by_technique = {}
    for technique, model_results in by_technique.items():
        if model_results:
            best_model = max(
                model_results.items(),
                key=lambda x: x[1].macro_f1
            )
            best_by_technique[technique] = {
                "model": best_model[0],
                "macro_f1": best_model[1].macro_f1,
                "macro_precision": best_model[1].macro_precision,
                "macro_recall": best_model[1].macro_recall,
            }
    
    # Find best technique for each model
    best_by_model = {}
    for model, technique_results in by_model.items():
        if technique_results:
            best_technique = max(
                technique_results.items(),
                key=lambda x: x[1].macro_f1
            )
            best_by_model[model] = {
                "technique": best_technique[0],
                "macro_f1": best_technique[1].macro_f1,
                "macro_precision": best_technique[1].macro_precision,
                "macro_recall": best_technique[1].macro_recall,
            }
    
    # Overall best combination
    best_overall = max(
        aggregated_results.items(),
        key=lambda x: x[1].macro_f1
    )
    
    return {
        "best_by_technique": best_by_technique,
        "best_by_model": best_by_model,
        "best_overall": {
            "combination": best_overall[0],
            "macro_f1": best_overall[1].macro_f1,
            "macro_precision": best_overall[1].macro_precision,
            "macro_recall": best_overall[1].macro_recall,
        },
        "by_technique": {
            tech: {
                model: {
                    "macro_f1": res.macro_f1,
                    "macro_precision": res.macro_precision,
                    "macro_recall": res.macro_recall,
                }
                for model, res in model_results.items()
            }
            for tech, model_results in by_technique.items()
        },
        "by_model": {
            model: {
                tech: {
                    "macro_f1": res.macro_f1,
                    "macro_precision": res.macro_precision,
                    "macro_recall": res.macro_recall,
                }
                for tech, res in tech_results.items()
            }
            for model, tech_results in by_model.items()
        }
    }


def _print_model_comparison(model_comparison: Dict[str, Any], logger) -> None:
    """Print model comparison analysis."""
    logger.info("\nBest Model for Each Technique:")
    logger.info("-" * 80)
    for technique, info in model_comparison["best_by_technique"].items():
        logger.info(
            f"  {technique:<8} -> {info['model']:<20} "
            f"(F1: {info['macro_f1']:.3f}, P: {info['macro_precision']:.3f}, R: {info['macro_recall']:.3f})"
        )
    
    logger.info("\nBest Technique for Each Model:")
    logger.info("-" * 80)
    for model, info in model_comparison["best_by_model"].items():
        logger.info(
            f"  {model:<20} -> {info['technique']:<8} "
            f"(F1: {info['macro_f1']:.3f}, P: {info['macro_precision']:.3f}, R: {info['macro_recall']:.3f})"
        )
    
    logger.info("\nBest Overall Combination:")
    logger.info("-" * 80)
    best = model_comparison["best_overall"]
    logger.info(
        f"  {best['combination']:<30} "
        f"(F1: {best['macro_f1']:.3f}, P: {best['macro_precision']:.3f}, R: {best['macro_recall']:.3f})"
    )


def _print_model_summaries(aggregated_results: Dict[str, AggregateResults], logger) -> None:
    """Print detailed summary for each model showing performance across all techniques."""
    # Group results by model
    by_model: Dict[str, Dict[str, AggregateResults]] = {}
    
    for combo_name, results in aggregated_results.items():
        # Parse combo_name (format: "technique_model")
        # Model names have '/' and '-' replaced with '_' in combo_name
        parts = combo_name.split("_", 1)
        if len(parts) == 2:
            technique = parts[0]
            # Reconstruct model name from combo_name format
            # combo_name format: technique_model where model has '/' and '-' replaced with '_'
            model_part = parts[1]
            
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
            
            if model not in by_model:
                by_model[model] = {}
            by_model[model][technique] = results
    
    # Sort models alphabetically for consistent output
    sorted_models = sorted(by_model.keys())
    
    for model in sorted_models:
        technique_results = by_model[model]
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Model: {model}")
        logger.info(f"{'=' * 80}")
        logger.info(f"Tested with {len(technique_results)} technique(s): {', '.join(sorted(technique_results.keys()))}")
        
        # Calculate averages across all techniques
        avg_f1 = sum(r.macro_f1 for r in technique_results.values()) / len(technique_results)
        avg_precision = sum(r.macro_precision for r in technique_results.values()) / len(technique_results)
        avg_recall = sum(r.macro_recall for r in technique_results.values()) / len(technique_results)
        avg_fuzzy_f1 = sum(r.fuzzy_macro_f1 for r in technique_results.values()) / len(technique_results)
        avg_exact_match = sum(r.avg_exact_match_rate for r in technique_results.values()) / len(technique_results)
        avg_graph_edit = sum(r.avg_graph_edit_distance for r in technique_results.values()) / len(technique_results)
        
        logger.info(f"\nAverage Performance Across All Techniques:")
        logger.info(f"  Macro F1:        {avg_f1:.3f}")
        logger.info(f"  Macro Precision: {avg_precision:.3f}")
        logger.info(f"  Macro Recall:    {avg_recall:.3f}")
        logger.info(f"  Fuzzy Macro F1:  {avg_fuzzy_f1:.3f}")
        logger.info(f"  Exact Match Rate: {avg_exact_match:.3f}")
        logger.info(f"  Graph Edit Dist: {avg_graph_edit:.2f}")
        
        # Find best and worst techniques for this model
        if len(technique_results) > 1:
            best_tech = max(technique_results.items(), key=lambda x: x[1].macro_f1)
            worst_tech = min(technique_results.items(), key=lambda x: x[1].macro_f1)
            
            logger.info(f"\nBest Technique: {best_tech[0]}")
            logger.info(f"  F1: {best_tech[1].macro_f1:.3f}, P: {best_tech[1].macro_precision:.3f}, R: {best_tech[1].macro_recall:.3f}")
            
            logger.info(f"\nWorst Technique: {worst_tech[0]}")
            logger.info(f"  F1: {worst_tech[1].macro_f1:.3f}, P: {worst_tech[1].macro_precision:.3f}, R: {worst_tech[1].macro_recall:.3f}")
        
        # Detailed breakdown by technique
        logger.info(f"\nPerformance by Technique:")
        logger.info("-" * 80)
        logger.info(f"{'Technique':<12} {'F1':<8} {'Precision':<10} {'Recall':<10} {'Fuzzy F1':<10} {'Exact Match':<12} {'Graph Edit':<12}")
        logger.info("-" * 80)
        
        # Sort techniques by F1 score (descending)
        sorted_techniques = sorted(technique_results.items(), key=lambda x: x[1].macro_f1, reverse=True)
        
        for technique, results in sorted_techniques:
            logger.info(
                f"{technique:<12} {results.macro_f1:<8.3f} {results.macro_precision:<10.3f} "
                f"{results.macro_recall:<10.3f} {results.fuzzy_macro_f1:<10.3f} "
                f"{results.avg_exact_match_rate:<12.3f} {results.avg_graph_edit_distance:<12.2f}"
            )
        
        # Additional metrics summary
        logger.info(f"\nAdditional Metrics Summary:")
        avg_omission = sum(r.avg_omission_rate for r in technique_results.values()) / len(technique_results)
        avg_hallucination = sum(r.avg_hallucination_rate for r in technique_results.values()) / len(technique_results)
        avg_redundancy = sum(r.avg_redundancy_rate for r in technique_results.values()) / len(technique_results)
        
        logger.info(f"  Average Omission Rate:      {avg_omission:.3f}")
        logger.info(f"  Average Hallucination Rate: {avg_hallucination:.3f}")
        logger.info(f"  Average Redundancy Rate:    {avg_redundancy:.3f}")
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Total Models Summarized: {len(sorted_models)}")
    logger.info(f"{'=' * 80}")


if __name__ == "__main__":
    # Test all prompting techniques with all models from figures
    # All models use OpenRouter API with provider prefix
    #         "openai/gpt-5-mini",                # OpenAI - faster, cost-efficient
    #   "openai/gpt-5-nano",                # OpenAI - smallest/fastest
    #   "google/gemini-2.0-flash-001",      # Google - Gemini 2.0 Flash
    #   "google/gemini-3-flash-preview",    # Google - Gemini 3 Flash Preview
    #   "meta-llama/llama-4-maverick",
    #   "deepseek/deepseek-chat-v3.1",      # DeepSeek - Chat v3.1
    #   "meta-llama/llama-3.1-70b-instruct",# Meta - Llama 3.1 70B
    #   "mistralai/mistral-nemo",           # Mistral - Nemo
    #   "anthropic/claude-sonnet-4.5",      # Anthropic - strong reasoning
    #   "openai/gpt-4.1",                   # OpenAI - smartest non-reasoning model
    #   "openai/gpt-4o",                    # OpenAI - GPT-4o


    # All models to test (from figures analysis)
    ALL_MODELS = [
        "openai/gpt-4o-mini",               # OpenAI - GPT-4o Mini
    ]

    # All techniques: baseline, improved, and full modes for IO, CoT, RAG, ReAct
    ALL_TECHNIQUES = [
        "Baseline-IO", "Improved-IO", "IO",
        "Baseline-CoT", "Improved-CoT", "CoT",
        "Baseline-RAG", "Improved-RAG", "RAG",
        "Baseline-ReAct", "Improved-ReAct", "ReAct",
    ]

    main(
        split="test",  # Use test split
        max_documents=2,  # Process two documents
        techniques=ALL_TECHNIQUES,
        models={technique: ALL_MODELS for technique in ALL_TECHNIQUES},
        max_workers=4,
        parallel_combinations=True,
        matching_strategies=["exact"],
    )
