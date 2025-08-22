"""Input validation utilities."""

from datetime import datetime
from typing import List

from ..models.document import Document, DocumentType
from ..models.query import Query
from ..core.exceptions import ValidationError


def validate_document(document: Document) -> None:
    """
    Validate document object.
    
    Args:
        document: Document to validate
        
    Raises:
        ValidationError: If document is invalid
    """
    try:
        if not isinstance(document, Document):
            raise ValidationError("Invalid document type")
        
        if not document.id or not document.id.strip():
            raise ValidationError("Document ID is required")
        
        if not document.content or not document.content.strip():
            raise ValidationError("Document content is required")
        
        if not document.entity_id or not document.entity_id.strip():
            raise ValidationError("Entity ID is required")
        
        if not isinstance(document.doc_type, DocumentType):
            raise ValidationError(f"Invalid document type: {document.doc_type}")
        
        if not isinstance(document.date, datetime):
            raise ValidationError("Document date must be a datetime object")
        
        # Check for reasonable date range (not too far in past/future)
        current_year = datetime.now().year
        if document.date.year < 1990 or document.date.year > current_year + 10:
            raise ValidationError(f"Document date year {document.date.year} is outside reasonable range")
        
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"Document validation failed: {str(e)}")


def validate_query(query: Query) -> None:
    """
    Validate query object.
    
    Args:
        query: Query to validate
        
    Raises:
        ValidationError: If query is invalid
    """
    try:
        if not isinstance(query, Query):
            raise ValidationError("Invalid query type")
        
        if not query.text or not query.text.strip():
            raise ValidationError("Query text is required")
        
        if not 0.0 <= query.similarity_threshold <= 1.0:
            raise ValidationError("Similarity threshold must be between 0.0 and 1.0")
        
        if query.max_results <= 0:
            raise ValidationError("Max results must be positive")
        
        if query.max_results > 1000:
            raise ValidationError("Max results cannot exceed 1000")
        
        # Validate entity IDs if provided
        if query.entity_ids:
            for entity_id in query.entity_ids:
                if not entity_id or not entity_id.strip():
                    raise ValidationError("Empty entity ID in filter")
        
        # Validate document types if provided
        if query.doc_types:
            for doc_type in query.doc_types:
                if not isinstance(doc_type, DocumentType):
                    raise ValidationError(f"Invalid document type in filter: {doc_type}")
        
        # Validate date range if provided
        if query.date_range:
            if query.date_range.start and query.date_range.end:
                if query.date_range.start > query.date_range.end:
                    raise ValidationError("Date range start must be before or equal to end")
        
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"Query validation failed: {str(e)}")


def validate_documents_batch(documents: List[Document]) -> None:
    """
    Validate a batch of documents.
    
    Args:
        documents: List of documents to validate
        
    Raises:
        ValidationError: If any document is invalid
    """
    if not documents:
        raise ValidationError("Document list cannot be empty")
    
    if len(documents) > 10000:
        raise ValidationError("Cannot process more than 10,000 documents in a single batch")
    
    # Check for duplicate IDs
    doc_ids = set()
    for i, doc in enumerate(documents):
        validate_document(doc)
        
        if doc.id in doc_ids:
            raise ValidationError(f"Duplicate document ID found: {doc.id}")
        doc_ids.add(doc.id)
