"""Registry for matcher strategies with factory pattern and auto-discovery."""

import logging
from typing import Dict, List, Optional, Type, Any
from pathlib import Path

from .base import BaseMatcher
from .exact_matcher import ExactMatcher
from .text_matcher_adapter import TextMatcherAdapter
from .fuzzy_matcher_adapter import FuzzyMatcherAdapter
from .jaccard_matcher import JaccardMatcher
from .token_matcher import TokenMatcher
from .levenshtein_matcher import LevenshteinMatcher
from .jaro_winkler_matcher import JaroWinklerMatcher

# Optional matchers (may not be available if dependencies missing)
try:
    from .bertscore_matcher_adapter import BERTScoreMatcherAdapter
    BERTSCORE_MATCHER_AVAILABLE = True
except ImportError:
    BERTSCORE_MATCHER_AVAILABLE = False
    BERTScoreMatcherAdapter = None

try:
    from .sbert_matcher import SBERTMatcher
    SBERT_MATCHER_AVAILABLE = True
except ImportError:
    SBERT_MATCHER_AVAILABLE = False
    SBERTMatcher = None

from .ensemble_matcher import EnsembleMatcher


class MatcherRegistry:
    """Registry for managing and creating matcher instances."""
    
    _matchers: Dict[str, Type[BaseMatcher]] = {}
    _default_configs: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        matcher_class: Type[BaseMatcher],
        default_config: Optional[Dict[str, Any]] = None
    ):
        """
        Register a matcher class.
        
        Args:
            name: Name identifier for the matcher
            matcher_class: Matcher class (must inherit from BaseMatcher)
            default_config: Optional default configuration dictionary
        """
        if not issubclass(matcher_class, BaseMatcher):
            raise ValueError(f"Matcher class must inherit from BaseMatcher: {matcher_class}")
        
        cls._matchers[name] = matcher_class
        cls._default_configs[name] = default_config or {}
    
    @classmethod
    def create(
        cls,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        entity_map=None,
        logger: Optional[logging.Logger] = None
    ) -> BaseMatcher:
        """
        Create a matcher instance by name.
        
        Args:
            name: Name of the matcher to create
            config: Optional configuration dictionary (merged with defaults)
            entity_map: Optional entity map (for matchers that need it)
            logger: Optional logger instance
            
        Returns:
            BaseMatcher instance
            
        Raises:
            ValueError: If matcher name is not registered
        """
        if name not in cls._matchers:
            available = ", ".join(cls._matchers.keys())
            raise ValueError(
                f"Matcher '{name}' not registered. Available matchers: {available}"
            )
        
        matcher_class = cls._matchers[name]
        default_config = cls._default_configs.get(name, {}).copy()
        
        # Merge provided config with defaults
        if config:
            default_config.update(config)
        
        # Add common parameters
        if logger:
            default_config['logger'] = logger
        
        # Special handling for matchers that need entity_map
        if entity_map and 'entity_map' in matcher_class.__init__.__code__.co_varnames:
            default_config['entity_map'] = entity_map
        
        try:
            return matcher_class(**default_config)
        except Exception as e:
            if logger:
                logger.error(f"Failed to create matcher '{name}': {e}")
            raise
    
    @classmethod
    def get_available_matchers(cls) -> List[str]:
        """
        Get list of available matcher names.
        
        Returns:
            List of registered matcher names
        """
        return list(cls._matchers.keys())
    
    @classmethod
    def is_available(cls, name: str) -> bool:
        """
        Check if a matcher is available.
        
        Args:
            name: Matcher name
            
        Returns:
            True if matcher is registered and available
        """
        return name in cls._matchers
    
    @classmethod
    def get_matcher_info(cls, name: str) -> Dict[str, Any]:
        """
        Get information about a matcher.
        
        Args:
            name: Matcher name
            
        Returns:
            Dictionary with matcher information
        """
        if name not in cls._matchers:
            return {}
        
        matcher_class = cls._matchers[name]
        default_config = cls._default_configs.get(name, {})
        
        return {
            "name": name,
            "class": matcher_class.__name__,
            "default_config": default_config,
            "module": matcher_class.__module__
        }


# Register all available matchers
def _register_all_matchers():
    """Register all available matcher classes."""
    
    # Always available matchers
    MatcherRegistry.register("exact", ExactMatcher, {
        "match_type": True,
        "similarity_threshold": 1.0
    })
    
    MatcherRegistry.register("text", TextMatcherAdapter, {
        "match_type": True,
        "similarity_threshold": 0.85,
        "fuzzy_threshold": 0.7,
        "use_bertscore": False
    })
    
    MatcherRegistry.register("fuzzy", FuzzyMatcherAdapter, {
        "match_type": True,
        "similarity_threshold": 0.7
    })
    
    MatcherRegistry.register("jaccard", JaccardMatcher, {
        "match_type": True,
        "similarity_threshold": 0.7
    })
    
    MatcherRegistry.register("token", TokenMatcher, {
        "match_type": True,
        "similarity_threshold": 0.7,
        "use_token_set": True
    })
    
    MatcherRegistry.register("levenshtein", LevenshteinMatcher, {
        "match_type": True,
        "similarity_threshold": 0.7
    })
    
    MatcherRegistry.register("jaro_winkler", JaroWinklerMatcher, {
        "match_type": True,
        "similarity_threshold": 0.7
    })
    
    # Optional matchers (if dependencies available)
    if BERTSCORE_MATCHER_AVAILABLE and BERTScoreMatcherAdapter:
        MatcherRegistry.register("bertscore", BERTScoreMatcherAdapter, {
            "match_type": True,
            "similarity_threshold": 0.85,
            "model_type": "microsoft/deberta-xlarge-mnli",
            "use_openai_embeddings": True
        })
    
    if SBERT_MATCHER_AVAILABLE and SBERTMatcher:
        MatcherRegistry.register("sbert", SBERTMatcher, {
            "match_type": True,
            "similarity_threshold": 0.7,
            "model_name": "all-MiniLM-L6-v2"
        })


# Auto-register on import
_register_all_matchers()


def create_matcher(
    name: str,
    config: Optional[Dict[str, Any]] = None,
    entity_map=None,
    logger: Optional[logging.Logger] = None
) -> BaseMatcher:
    """
    Convenience function to create a matcher.
    
    Args:
        name: Matcher name
        config: Optional configuration
        entity_map: Optional entity map
        logger: Optional logger
        
    Returns:
        BaseMatcher instance
    """
    return MatcherRegistry.create(name, config, entity_map, logger)


def get_available_matchers() -> List[str]:
    """
    Get list of available matcher names.
    
    Returns:
        List of matcher names
    """
    return MatcherRegistry.get_available_matchers()

