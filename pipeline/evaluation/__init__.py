"""Evaluation components."""

from .evaluator import Evaluator
from .matcher import RelationMatcher
from .text_matcher import TextRelationMatcher
from .metrics import MetricsCalculator
from .strategy_comparison import compare_matching_strategies
from .document_exporter import DocumentExporter
from .matchers.registry import MatcherRegistry, create_matcher, get_available_matchers

# Try to import BERTScore matcher (optional dependency)
try:
    from .bertscore_matcher import BERTScoreMatcher
    __all__ = [
        "Evaluator",
        "RelationMatcher",
        "TextRelationMatcher",
        "MetricsCalculator",
        "BERTScoreMatcher",
        "compare_matching_strategies",
        "DocumentExporter",
        "MatcherRegistry",
        "create_matcher",
        "get_available_matchers",
    ]
except ImportError:
    __all__ = [
        "Evaluator",
        "RelationMatcher",
        "TextRelationMatcher",
        "MetricsCalculator",
        "compare_matching_strategies",
        "DocumentExporter",
        "MatcherRegistry",
        "create_matcher",
        "get_available_matchers",
    ]
