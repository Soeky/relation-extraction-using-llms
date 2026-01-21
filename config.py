"""Configuration management."""

import os
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for the pipeline."""
    
    # API Configuration
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    # OpenAI API Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", None)
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    # NCBI/PubMed API Configuration
    NCBI_API_KEY: Optional[str] = os.getenv("NCBI_API_KEY", None)
    NCBI_EMAIL: Optional[str] = os.getenv("NCBI_EMAIL", None)
    
    # Model Configuration - can be changed per technique
    DEFAULT_MODEL: str = "openai/gpt-4o-mini"
    
    # Available models for testing (OpenRouter compatible)
    AVAILABLE_MODELS: Dict[str, str] = {
        # OpenAI Models - GPT-5 Series (Latest)
        "gpt-5.1": "openai/gpt-5.1",  # Best for coding and agentic tasks with configurable reasoning effort
        "gpt-5": "openai/gpt-5",  # Previous intelligent reasoning model for coding and agentic tasks
        "gpt-5-mini": "openai/gpt-5-mini",  # Faster, cost-efficient version for well-defined tasks
        "gpt-5-nano": "openai/gpt-5-nano",  # Fastest, most cost-efficient version
        
        # OpenAI Models - GPT-4 Series
        "gpt-4.1": "openai/gpt-4.1",  # Smartest non-reasoning model
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "gpt-4o": "openai/gpt-4o",
        "gpt-4o-2024-08-06": "openai/gpt-4o-2024-08-06",
        "gpt-4o-2024-11-20": "openai/gpt-4o-2024-11-20",
        "gpt-4-turbo": "openai/gpt-4-turbo",
        "gpt-4-turbo-2024-04-09": "openai/gpt-4-turbo-2024-04-09",
        "gpt-4-0125-preview": "openai/gpt-4-0125-preview",
        "gpt-4-1106-preview": "openai/gpt-4-1106-preview",
        "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
        "gpt-3.5-turbo-16k": "openai/gpt-3.5-turbo-16k",
        
        # OpenAI Models - O1 Series (Reasoning) - Removed: models not accessible
        
        # Anthropic Claude Models
        "claude-sonnet-4.5": "anthropic/claude-sonnet-4.5",
        "claude-sonnet-4": "anthropic/claude-sonnet-4",
        "claude-opus-4.5": "anthropic/claude-opus-4.5",
        "claude-haiku-4.5": "anthropic/claude-haiku-4.5",
        "claude-3.7-sonnet": "anthropic/claude-3.7-sonnet",
        "claude-3.5-haiku": "anthropic/claude-3.5-haiku",
        "claude-3.7-sonnet-thinking": "anthropic/claude-3.7-sonnet:thinking",
        
        # Meta Llama Models
        "llama-4-maverick": "meta-llama/llama-4-maverick",
        "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct",
        "llama-4-scout": "meta-llama/llama-4-scout",
        "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
        "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
        "llama-3.2-3b": "meta-llama/llama-3.2-3b-instruct",
        
        # Google Gemini Models
        "gemini-2.0-flash": "google/gemini-2.0-flash-001",  # Valid OpenRouter model ID
        "gemini-2.0-flash-lite": "google/gemini-2.0-flash-lite-001",  # Valid OpenRouter model ID
        "gemini-2.5-flash": "google/gemini-2.5-flash",  # Latest Gemini 2.5 Flash
        "gemini-2.5-flash-lite": "google/gemini-2.5-flash-lite",  # Lite version
        "gemini-2.5-pro": "google/gemini-2.5-pro",  # Gemini 2.5 Pro
        
        # Mistral AI Models
        "mistral-nemo": "mistralai/mistral-nemo",
        "mistral-small-3.2-24b": "mistralai/mistral-small-3.2-24b-instruct",
        "mistral-small-24b-2501": "mistralai/mistral-small-24b-instruct-2501",
        "codestral-2508": "mistralai/codestral-2508",
        "mistral-small-3.1-24b": "mistralai/mistral-small-3.1-24b-instruct",
        "mistral-medium-3.1": "mistralai/mistral-medium-3.1",
        
        # DeepSeek Models
        "deepseek-chat-v3-0324": "deepseek/deepseek-chat-v3-0324",
        "deepseek-chat-v3.1": "deepseek/deepseek-chat-v3.1",
        "deepseek-v3.2-exp": "deepseek/deepseek-v3.2-exp",
        "deepseek-r1-0528": "deepseek/deepseek-r1-0528",
        "deepseek-v3.1-terminus": "deepseek/deepseek-v3.1-terminus",
        
        # Qwen Models
        "qwen3-next-80b": "qwen/qwen3-next-80b-a3b-instruct",
        "qwen3-235b-2507": "qwen/qwen3-235b-a22b-2507",
        "qwen3-vl-235b": "qwen/qwen3-vl-235b-a22b-instruct",
        "qwen3-coder": "qwen/qwen3-coder",
        
        # Other Models
        "pixtral-12b": "mistralai/pixtral-12b",
        "sonar-pro": "perplexity/sonar-pro",
        "sonar": "perplexity/sonar",
    }
    
    # Data paths
    BASE_PATH: Path = Path(__file__).parent
    CLEAN_TEXT_PATH: Path = BASE_PATH / "clean_text"
    GOLD_RELATIONS_PATH: Path = BASE_PATH / "gold_relations"
    OUTPUT_DIR: Path = BASE_PATH / "results"
    LLM_CACHE_DIR: Path = BASE_PATH / "llm_cache"  # Directory for cached LLM responses
    
    # RAG Configuration
    RAG_SOURCE_DIR: Path = BASE_PATH / "rag_sources"  # Directory for source files
    RAG_EMBEDDINGS_DIR: Path = BASE_PATH / "rag_embeddings"  # Directory for cached embeddings
    RAG_EMBEDDING_MODEL: str = "text-embedding-3-small"  # OpenAI embedding model
    
    # BERTScore Configuration
    BERTSCORE_MODEL: str = "microsoft/deberta-xlarge-mnli"  # BERTScore model
    USE_BERTSCORE: bool = False  # Whether to use BERTScore for matching
    BERTSCORE_THRESHOLD: float = 0.70  # Minimum BERTScore similarity for a match
    USE_OPENAI_EMBEDDINGS_FOR_BERTSCORE: bool = True  # Use OpenAI embeddings API instead of BERTScore (much faster)
    RAG_TOP_K: int = 5  # Number of retrieved documents
    
    # LLM Configuration
    MAX_TOKENS: int = 10000  # Default max tokens for most models
    MAX_TOKENS_GPT5: int = 15000  # GPT-5 models need more tokens (reasoning tokens count as output)
    TEMPERATURE: float = 0.0
    INCLUDE_RELATION_TYPES: bool = False  # Whether to include relation type definitions in prompts
    
    # Evaluation Configuration
    # NOTE: "text" strategy converts gold relations (IDs) to text mentions and compares text-to-text
    # This avoids entity resolution issues and is the recommended approach
    # NOTE: These are DEFAULT values. main.py runs ALL strategies by dynamically setting these values.
    # These defaults only matter if Evaluator is used directly (e.g., in scripts).
    MATCHING_STRATEGY: str = "text"  # "exact", "fuzzy", "bertscore", "text", "jaccard", "token", "levenshtein", "jaro_winkler", "sbert" - default only, main.py overrides
    USE_BERTSCORE: bool = False  # Whether to use BERTScore for semantic matching - default only, main.py overrides
    BERTSCORE_THRESHOLD: float = 0.70  # Minimum BERTScore similarity for a match (0-1) - lowered from 0.85 to allow more matches
    BERTSCORE_MODEL: str = "microsoft/deberta-xlarge-mnli"  # BERT model for BERTScore
    USE_OPENAI_EMBEDDINGS_FOR_BERTSCORE: bool = True  # Use OpenAI embeddings API instead of BERTScore (much faster, requires OPENAI_API_KEY)
    TEXT_MATCHING_FUZZY_THRESHOLD: float = 0.7  # Minimum fuzzy string similarity for text matching (0-1)
    
    # Matcher Configuration
    # Thresholds for different matching strategies (minimum 70% to maintain quality)
    JACCARD_THRESHOLD: float = 0.7  # Minimum Jaccard similarity for jaccard matcher
    TOKEN_THRESHOLD: float = 0.7  # Minimum token similarity for token matcher
    TOKEN_USE_TOKEN_SET: bool = True  # Use token_set_ratio (True) or token_sort_ratio (False) for token matcher
    LEVENSHTEIN_THRESHOLD: float = 0.7  # Minimum Levenshtein similarity
    JARO_WINKLER_THRESHOLD: float = 0.7  # Minimum Jaro-Winkler similarity
    SBERT_THRESHOLD: float = 0.7  # Minimum SBERT cosine similarity
    SBERT_MODEL: str = "all-MiniLM-L6-v2"  # SBERT model name (e.g., "all-MiniLM-L6-v2", "all-mpnet-base-v2")
    
    # LLM Cache Configuration
    USE_LLM_CACHE: bool = False  # Whether to use cached LLM responses
    SAVE_LLM_CACHE: bool = True  # Whether to save LLM responses to cache
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR" - Set to DEBUG to see all logs including raw LLM responses
    LOG_TO_FILE: bool = True
    LOG_TO_CONSOLE: bool = True
    
    @classmethod
    def validate(cls) -> None:
        """Validate configuration."""
        # At least one API key must be present
        if not cls.OPENROUTER_API_KEY and not cls.OPENAI_API_KEY:
            raise ValueError("At least one of OPENROUTER_API_KEY or OPENAI_API_KEY must be set in environment variables")
        
        # Create necessary directories
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        cls.RAG_SOURCE_DIR.mkdir(exist_ok=True)
        cls.RAG_EMBEDDINGS_DIR.mkdir(exist_ok=True)
        if cls.USE_LLM_CACHE or cls.SAVE_LLM_CACHE:
            cls.LLM_CACHE_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def get_model_name(cls, model_key: Optional[str] = None) -> str:
        """
        Get model name from key or return default.

        Args:
            model_key: Key from AVAILABLE_MODELS, full model name (with provider/), or None for default

        Returns:
            Full model name (with provider prefix for OpenRouter, or plain name for OpenAI)
        """
        if not model_key:
            return cls.DEFAULT_MODEL

        # If model_key is already a full model name (contains "/"), return it as-is
        if "/" in model_key:
            return model_key

        # Otherwise, look up in AVAILABLE_MODELS
        if model_key in cls.AVAILABLE_MODELS:
            return cls.AVAILABLE_MODELS[model_key]

        return cls.DEFAULT_MODEL
    
    @classmethod
    def is_openai_model(cls, model_name: str) -> bool:
        """
        Check if a model is an OpenAI model that can be used directly via OpenAI API.
        
        GPT-5.x models use the OpenAI API but with the /responses endpoint instead of /chat/completions.
        
        Args:
            model_name: Full model name (e.g., "openai/gpt-4o" or "gpt-4o")
            
        Returns:
            True if the model is an OpenAI model that can be used via OpenAI API directly
        """
        # Check if it starts with "openai/" or just "gpt-"
        return model_name.startswith("openai/") or model_name.startswith("gpt-")
    
    @classmethod
    def get_openai_model_name(cls, model_name: str) -> str:
        """
        Get the OpenAI model name (strip "openai/" prefix if present).
        
        Args:
            model_name: Full model name (e.g., "openai/gpt-4o")
            
        Returns:
            OpenAI model name (e.g., "gpt-4o")
        """
        if model_name.startswith("openai/"):
            return model_name[7:]  # Remove "openai/" prefix
        return model_name
    
    @classmethod
    def requires_max_completion_tokens(cls, model_name: str) -> bool:
        """
        Check if a model requires 'max_completion_tokens' instead of 'max_tokens'.
        
        GPT-5.x and GPT-4.1 models require 'max_completion_tokens'.
        Older models use 'max_tokens'.
        
        Args:
            model_name: Full model name (e.g., "openai/gpt-5.1" or "gpt-4o")
            
        Returns:
            True if model requires 'max_completion_tokens', False for 'max_tokens'
        """
        model_lower = model_name.lower()
        # Check for GPT-5 series
        if any(gpt5 in model_lower for gpt5 in ["gpt-5", "gpt-5.1", "gpt-5-pro", "gpt-5-mini", "gpt-5-nano"]):
            return True
        # Check for GPT-4.1
        if "gpt-4.1" in model_lower:
            return True
        # Check for Claude 4.x models (require max_completion_tokens)
        # These include claude-sonnet-4, claude-sonnet-4.5, claude-opus-4.5, claude-haiku-4.5, claude-3.7-sonnet
        if any(claude4 in model_lower for claude4 in ["claude-sonnet-4", "claude-opus-4", "claude-haiku-4", "claude-3.7"]):
            return True
        return False
    
    @classmethod
    def requires_default_temperature(cls, model_name: str) -> bool:
        """
        Check if a model requires default temperature (1.0) and doesn't support custom temperature.
        
        GPT-5.x models only support the default temperature value (1.0).
        Setting temperature=0.0 or any other value will cause an error.
        
        Args:
            model_name: Full model name (e.g., "openai/gpt-5.1" or "gpt-4o")
            
        Returns:
            True if model requires default temperature (should omit temperature parameter),
            False if custom temperature is supported
        """
        model_lower = model_name.lower()
        # Check for GPT-5 series - these only support default temperature
        if any(gpt5 in model_lower for gpt5 in ["gpt-5", "gpt-5.1", "gpt-5-pro", "gpt-5-mini", "gpt-5-nano"]):
            return True
        return False
    
    @classmethod
    def get_max_tokens_for_model(cls, model_name: str) -> int:
        """
        Get the appropriate max tokens value for a model.
        
        GPT-5.x models need more tokens because reasoning tokens count as output tokens.
        
        Args:
            model_name: Full model name (e.g., "openai/gpt-5.1" or "gpt-4o")
            
        Returns:
            Max tokens value to use for this model
        """
        model_lower = model_name.lower()
        # Check for GPT-5 series - these need more tokens
        if any(gpt5 in model_lower for gpt5 in ["gpt-5", "gpt-5.1", "gpt-5-pro", "gpt-5-mini", "gpt-5-nano"]):
            return cls.MAX_TOKENS_GPT5
        return cls.MAX_TOKENS
    
    @classmethod
    def requires_responses_endpoint(cls, model_name: str) -> bool:
        """
        Check if a model requires the /responses endpoint instead of /chat/completions.
        
        GPT-5.x models use the newer Responses API endpoint (/responses).
        Older models use the Chat Completions API endpoint (/chat/completions).
        
        Args:
            model_name: Full model name (e.g., "openai/gpt-5.1" or "gpt-4o")
            
        Returns:
            True if model requires /responses endpoint, False for /chat/completions
        """
        model_lower = model_name.lower()
        # Check for GPT-5 series - these use the /responses endpoint
        if any(gpt5 in model_lower for gpt5 in ["gpt-5", "gpt-5.1", "gpt-5-pro", "gpt-5-mini", "gpt-5-nano"]):
            return True
        return False
