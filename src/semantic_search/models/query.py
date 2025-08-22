"""Query data model with filtering capabilities."""

from datetime import datetime
from typing import Optional, Set
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator

from .document import DocumentType


@dataclass
class DateRange:
    """Date range filter for queries."""
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    
    def __post_init__(self) -> None:
        """Validate date range."""
        if self.start and self.end and self.start > self.end:
            raise ValueError("Start date must be before or equal to end date")
    
    def contains(self, date: datetime) -> bool:
        """Check if date falls within range."""
        if self.start and date < self.start:
            return False
        if self.end and date > self.end:
            return False
        return True


@dataclass
class Query:
    """
    Search query with filtering and configuration options.
    
    Attributes:
        text: Search text query
        entity_ids: Filter by specific entity IDs (None = all entities)
        doc_types: Filter by document types (None = all types)
        date_range: Filter by date range (None = all dates)
        similarity_threshold: Minimum similarity score (0.0-1.0)
        max_results: Maximum number of results to return
    """
    text: str
    entity_ids: Optional[Set[str]] = None
    doc_types: Optional[Set[DocumentType]] = None
    date_range: Optional[DateRange] = None
    similarity_threshold: float = 0.1
    max_results: int = 50
    
    def __post_init__(self) -> None:
        """Validate query parameters."""
        if not self.text.strip():
            raise ValueError("Query text cannot be empty")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
        if self.max_results <= 0:
            raise ValueError("Max results must be positive")
        if self.max_results > 1000:
            raise ValueError("Max results cannot exceed 1000")


class QueryModel(BaseModel):
    """Pydantic model for query validation in API contexts."""
    
    text: str = Field(..., min_length=1, description="Search query text")
    entity_ids: Optional[Set[str]] = Field(None, description="Entity ID filters")
    doc_types: Optional[Set[DocumentType]] = Field(None, description="Document type filters")
    date_range: Optional[DateRange] = Field(None, description="Date range filter")
    similarity_threshold: float = Field(0.1, ge=0.0, le=1.0, description="Minimum similarity score")
    max_results: int = Field(50, ge=1, le=1000, description="Maximum results to return")
    
    @validator('text')
    def validate_text(cls, v: str) -> str:
        """Ensure query text is not just whitespace."""
        if not v.strip():
            raise ValueError('Query text cannot be empty or whitespace only')
        return v.strip()
    
    def to_query(self) -> Query:
        """Convert to Query dataclass."""
        return Query(
            text=self.text,
            entity_ids=self.entity_ids,
            doc_types=self.doc_types,
            date_range=self.date_range,
            similarity_threshold=self.similarity_threshold,
            max_results=self.max_results
        )
