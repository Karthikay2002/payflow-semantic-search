"""Search result data model."""

from typing import List, Dict, Any
from dataclasses import dataclass

from .document import Document


@dataclass
class SearchResult:
    """
    Search result with relevance scoring and context.
    
    Attributes:
        document: The matched document
        score: Similarity score (0.0-1.0, higher is better)
        context_snippet: Relevant text snippet from document
        matched_terms: Terms that contributed to the match
        rank: Result ranking position (1-based)
    """
    document: Document
    score: float
    context_snippet: str
    matched_terms: List[str]
    rank: int
    
    def __post_init__(self) -> None:
        """Validate search result."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("Score must be between 0.0 and 1.0")
        if self.rank <= 0:
            raise ValueError("Rank must be positive")
        if not self.context_snippet.strip():
            raise ValueError("Context snippet cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "document": {
                "id": self.document.id,
                "entity_id": self.document.entity_id,
                "doc_type": self.document.doc_type.value,
                "date": self.document.date.isoformat(),
                "metadata": self.document.metadata
            },
            "score": round(self.score, 4),
            "context_snippet": self.context_snippet,
            "matched_terms": self.matched_terms,
            "rank": self.rank
        }
