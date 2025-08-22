"""Data models for semantic search system."""

from .document import Document, DocumentType
from .query import Query, DateRange
from .result import SearchResult

__all__ = ["Document", "DocumentType", "Query", "DateRange", "SearchResult"]
