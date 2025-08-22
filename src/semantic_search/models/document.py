"""Document data model with validation."""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, validator, Field


class DocumentType(str, Enum):
    """Supported financial document types."""
    INVOICE = "invoice"
    PURCHASE_ORDER = "purchase_order"
    CONTRACT = "contract"
    RECEIPT = "receipt"
    STATEMENT = "statement"
    # SEC Filing Types
    ANNUAL_REPORT = "annual_report"  # 10-K
    QUARTERLY_REPORT = "quarterly_report"  # 10-Q
    CURRENT_REPORT = "current_report"  # 8-K
    PROXY_STATEMENT = "proxy_statement"  # DEF 14A
    # Earnings and News
    EARNINGS_TRANSCRIPT = "earnings_transcript"
    NEWS_ARTICLE = "news_article"
    MARKET_ANALYSIS = "market_analysis"
    RESEARCH_REPORT = "research_report"
    # Other Financial Documents
    FINANCIAL_STATEMENT = "financial_statement"
    PRESS_RELEASE = "press_release"
    OTHER = "other"


@dataclass
class Document:
    """
    Financial document with metadata for semantic search.
    
    Attributes:
        id: Unique document identifier
        content: Full text content of the document
        entity_id: Entity/company identifier
        doc_type: Type of financial document
        date: Document creation/issue date
        metadata: Additional document metadata
    """
    id: str
    content: str
    entity_id: str
    doc_type: DocumentType
    date: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate document after initialization."""
        if not self.id.strip():
            raise ValueError("Document ID cannot be empty")
        if not self.content.strip():
            raise ValueError("Document content cannot be empty")
        if not self.entity_id.strip():
            raise ValueError("Entity ID cannot be empty")
        if not isinstance(self.doc_type, DocumentType):
            raise ValueError(f"Invalid document type: {self.doc_type}")


class DocumentModel(BaseModel):
    """Pydantic model for document validation in API contexts."""
    
    id: str = Field(..., min_length=1, description="Unique document identifier")
    content: str = Field(..., min_length=1, description="Document text content")
    entity_id: str = Field(..., min_length=1, description="Entity identifier")
    doc_type: DocumentType = Field(..., description="Document type")
    date: datetime = Field(..., description="Document date")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('content')
    def validate_content(cls, v: str) -> str:
        """Ensure content is not just whitespace."""
        if not v.strip():
            raise ValueError('Content cannot be empty or whitespace only')
        return v.strip()
    
    @validator('entity_id')
    def validate_entity_id(cls, v: str) -> str:
        """Ensure entity_id is valid."""
        if not v.strip():
            raise ValueError('Entity ID cannot be empty or whitespace only')
        return v.strip()
    
    def to_document(self) -> Document:
        """Convert to Document dataclass."""
        return Document(
            id=self.id,
            content=self.content,
            entity_id=self.entity_id,
            doc_type=self.doc_type,
            date=self.date,
            metadata=self.metadata
        )
