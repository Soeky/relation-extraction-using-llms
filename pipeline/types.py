"""Type definitions for the relation extraction pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Mention:
    """A single mention of an entity in text."""
    text: str
    passage_index: int
    passage_offset: int
    char_offset: int
    length: int


@dataclass
class Entity:
    """An entity with its mentions in a document."""
    id: str  # Global entity identifier (e.g., "D003409", "6528")
    type: str  # Entity type (e.g., "DiseaseOrPhenotypicFeature", "GeneOrGeneProduct")
    mentions: List[Mention] = field(default_factory=list)


@dataclass
class Relation:
    """A relation between two entities."""
    id: str
    head_id: str  # Entity ID of the head entity
    tail_id: str  # Entity ID of the tail entity
    type: str  # Relation type (e.g., "Association", "Positive_Correlation")
    novel: str = "No"  # "Novel" or "No"


@dataclass
class Document:
    """A document with its text content."""
    doc_id: str
    text: str  # Full text (title + body)
    title: Optional[str] = None
    body: Optional[str] = None


@dataclass
class GoldRelations:
    """Gold standard relations for a document."""
    doc_id: str
    entities: List[Entity]
    relations: List[Relation]
    title: Optional[str] = None
    body: Optional[str] = None
    file_path: Optional[str] = None  # Path to the gold relations JSON file


@dataclass
class GlobalEntity:
    """A global entity aggregated across all documents."""
    id: str
    type: str
    all_mentions: List[Mention] = field(default_factory=list)
    common_mentions: List[str] = field(default_factory=list)  # Most frequent surface forms
    document_count: int = 0  # Number of documents containing this entity
    canonical_name: str = ""  # Most common mention text


@dataclass
class ParsedRelation:
    """A relation extracted from LLM response."""
    head_mention: str  # Text mention of head entity
    tail_mention: str  # Text mention of tail entity
    relation_type: str
    head_id: Optional[str] = None  # Resolved entity ID
    tail_id: Optional[str] = None  # Resolved entity ID
    confidence: Optional[float] = None


@dataclass
class ParsedRelations:
    """Parsed relations from LLM response."""
    relations: List[ParsedRelation] = field(default_factory=list)
    entities: Optional[List[Entity]] = None
    confidence_scores: Optional[List[float]] = None
    parsing_errors: List[str] = field(default_factory=list)
    entity_resolution_errors: List[str] = field(default_factory=list)
    doc_id: Optional[str] = None  # Document ID for tracking


@dataclass
class EvaluationResult:
    """Evaluation results for a single document."""
    doc_id: str
    strategy: str = "unknown"  # Matching strategy name
    true_positives: List[Relation] = field(default_factory=list)
    false_positives: List[ParsedRelation] = field(default_factory=list)  # FPs are ParsedRelations
    false_negatives: List[Relation] = field(default_factory=list)
    partial_matches: List[tuple] = field(default_factory=list)  # (ParsedRelation, Relation) pairs
    semantic_matches: List[tuple] = field(default_factory=list)  # (ParsedRelation, Relation, similarity_score) tuples
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    # Fuzzy metrics (treating partial matches as correct for entities)
    fuzzy_precision: float = 0.0
    fuzzy_recall: float = 0.0
    fuzzy_f1: float = 0.0
    exact_match_rate: float = 0.0
    omission_rate: float = 0.0
    hallucination_rate: float = 0.0
    redundancy_rate: float = 0.0
    graph_edit_distance: float = 0.0
    bertscore: float = 0.0
    per_type_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Detailed match information for JSON export
    detailed_matches: List[Dict[str, Any]] = field(default_factory=list)  # Per-match details with scores


@dataclass
class AggregateResults:
    """Aggregated results across all documents for a technique."""
    technique_name: str
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    micro_precision: float = 0.0
    micro_recall: float = 0.0
    micro_f1: float = 0.0
    avg_exact_match_rate: float = 0.0
    avg_omission_rate: float = 0.0
    avg_hallucination_rate: float = 0.0
    avg_redundancy_rate: float = 0.0
    avg_graph_edit_distance: float = 0.0
    avg_bertscore: float = 0.0
    # Total counts across all documents
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    total_gold: int = 0  # total_tp + total_fn
    total_predicted: int = 0  # total_tp + total_fp
    # Graph Edit Distance (aggregated)
    total_graph_edit_distance: float = 0.0  # Sum of GED across all documents
    normalized_graph_edit_distance: float = 0.0  # total_GED / total_gold (lower is better)
    # Overall aggregated rates (calculated from totals, not averaged)
    overall_exact_match_rate: float = 0.0  # total_tp / total_gold
    overall_omission_rate: float = 0.0  # total_fn / total_gold
    overall_hallucination_rate: float = 0.0  # total_fp / total_predicted
    # Fuzzy/partial match statistics
    total_partial_matches: int = 0
    avg_partial_matches: float = 0.0
    # Fuzzy micro averages (calculated from aggregated TP/FP/FN including partial matches)
    fuzzy_micro_precision: float = 0.0
    fuzzy_micro_recall: float = 0.0
    fuzzy_micro_f1: float = 0.0
    # Fuzzy macro averages (average of per-document fuzzy metrics)
    fuzzy_macro_precision: float = 0.0
    fuzzy_macro_recall: float = 0.0
    fuzzy_macro_f1: float = 0.0
    per_document_results: List[EvaluationResult] = field(default_factory=list)
    # Statistical spread metrics for F1
    f1_std: float = 0.0  # Standard deviation
    f1_median: float = 0.0
    f1_min: float = 0.0
    f1_max: float = 0.0
    # Statistical spread metrics for Precision
    precision_std: float = 0.0
    precision_median: float = 0.0
    precision_min: float = 0.0
    precision_max: float = 0.0
    # Statistical spread metrics for Recall
    recall_std: float = 0.0
    recall_median: float = 0.0
    recall_min: float = 0.0
    recall_max: float = 0.0

