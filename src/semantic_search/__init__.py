"""
Semantic Search Utility for Financial Documents

A production-ready semantic search system that processes and indexes financial 
documents (invoices, purchase orders) with TF-IDF embeddings and vector similarity search.
"""

from .api.service import SemanticSearchService
from .models.document import Document, DocumentType
from .models.query import Query, DateRange
from .models.result import SearchResult
from .core.engine import SemanticSearchEngine

__version__ = "1.0.0"
__author__ = "Karthikay Gundepudi"

__all__ = [
    "SemanticSearchService",
    "SemanticSearchEngine", 
    "Document",
    "DocumentType",
    "Query",
    "DateRange",
    "SearchResult",
]
