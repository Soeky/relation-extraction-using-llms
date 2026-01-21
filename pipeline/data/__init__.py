"""Data loading and entity management components."""

from .loader import DocumentLoader, GoldRelationsLoader, DatasetLoader
from .entity_map import GlobalEntityMap

__all__ = [
    "DocumentLoader",
    "GoldRelationsLoader",
    "DatasetLoader",
    "GlobalEntityMap",
]
