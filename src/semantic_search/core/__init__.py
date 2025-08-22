"""Core engine components for semantic search."""

from .engine import SemanticSearchEngine
from .embeddings import TFIDFEmbedding
from .exceptions import (
    SemanticSearchError,
    DocumentProcessingError,
    IndexError,
    SearchError,
    ValidationError
)

__all__ = [
    "SemanticSearchEngine",
    "TFIDFEmbedding",
    "SemanticSearchError",
    "DocumentProcessingError", 
    "IndexError",
    "SearchError",
    "ValidationError"
]
