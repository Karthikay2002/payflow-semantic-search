"""Test data models and validation."""

import pytest
from datetime import datetime, timedelta
from typing import Set

from semantic_search.models.document import Document, DocumentType, DocumentModel
from semantic_search.models.query import Query, QueryModel, DateRange
from semantic_search.models.result import SearchResult


class TestDocument:
    """Test Document model."""
    
    def test_valid_document_creation(self):
        """Test creating a valid document."""
        doc = Document(
            id="test_001",
            content="Test invoice content",
            entity_id="test_entity",
            doc_type=DocumentType.INVOICE,
            date=datetime.now(),
            metadata={"amount": 100.0}
        )
        
        assert doc.id == "test_001"
        assert doc.content == "Test invoice content"
        assert doc.entity_id == "test_entity"
        assert doc.doc_type == DocumentType.INVOICE
        assert doc.metadata["amount"] == 100.0
    
    def test_empty_id_validation(self):
        """Test validation for empty document ID."""
        with pytest.raises(ValueError, match="Document ID cannot be empty"):
            Document(
                id="",
                content="Test content",
                entity_id="test_entity",
                doc_type=DocumentType.INVOICE,
                date=datetime.now()
            )
    
    def test_empty_content_validation(self):
        """Test validation for empty content."""
        with pytest.raises(ValueError, match="Document content cannot be empty"):
            Document(
                id="test_001",
                content="",
                entity_id="test_entity",
                doc_type=DocumentType.INVOICE,
                date=datetime.now()
            )
    
    def test_empty_entity_id_validation(self):
        """Test validation for empty entity ID."""
        with pytest.raises(ValueError, match="Entity ID cannot be empty"):
            Document(
                id="test_001",
                content="Test content",
                entity_id="",
                doc_type=DocumentType.INVOICE,
                date=datetime.now()
            )
    
    def test_invalid_doc_type_validation(self):
        """Test validation for invalid document type."""
        with pytest.raises(ValueError, match="Invalid document type"):
            Document(
                id="test_001",
                content="Test content",
                entity_id="test_entity",
                doc_type="invalid_type",  # type: ignore
                date=datetime.now()
            )


class TestDocumentModel:
    """Test DocumentModel pydantic validation."""
    
    def test_valid_document_model(self):
        """Test valid document model creation."""
        model = DocumentModel(
            id="test_001",
            content="Test invoice content",
            entity_id="test_entity",
            doc_type=DocumentType.INVOICE,
            date=datetime.now(),
            metadata={"amount": 100.0}
        )
        
        doc = model.to_document()
        assert isinstance(doc, Document)
        assert doc.id == "test_001"
    
    def test_whitespace_content_validation(self):
        """Test validation for whitespace-only content."""
        with pytest.raises(ValueError, match="Content cannot be empty"):
            DocumentModel(
                id="test_001",
                content="   ",  # Only whitespace
                entity_id="test_entity",
                doc_type=DocumentType.INVOICE,
                date=datetime.now()
            )


class TestDateRange:
    """Test DateRange model."""
    
    def test_valid_date_range(self):
        """Test valid date range creation."""
        start = datetime.now() - timedelta(days=30)
        end = datetime.now()
        
        date_range = DateRange(start=start, end=end)
        assert date_range.start == start
        assert date_range.end == end
    
    def test_invalid_date_range(self):
        """Test invalid date range (start > end)."""
        start = datetime.now()
        end = datetime.now() - timedelta(days=30)
        
        with pytest.raises(ValueError, match="Start date must be before or equal to end date"):
            DateRange(start=start, end=end)
    
    def test_date_contains(self):
        """Test date range contains method."""
        start = datetime.now() - timedelta(days=30)
        end = datetime.now()
        date_range = DateRange(start=start, end=end)
        
        # Date within range
        test_date = datetime.now() - timedelta(days=15)
        assert date_range.contains(test_date)
        
        # Date before range
        test_date = datetime.now() - timedelta(days=60)
        assert not date_range.contains(test_date)
        
        # Date after range
        test_date = datetime.now() + timedelta(days=1)
        assert not date_range.contains(test_date)
    
    def test_open_date_range(self):
        """Test date range with only start or end."""
        # Only start date
        start = datetime.now() - timedelta(days=30)
        date_range = DateRange(start=start, end=None)
        
        assert date_range.contains(datetime.now())
        assert not date_range.contains(datetime.now() - timedelta(days=60))
        
        # Only end date
        end = datetime.now()
        date_range = DateRange(start=None, end=end)
        
        assert date_range.contains(datetime.now() - timedelta(days=30))
        assert not date_range.contains(datetime.now() + timedelta(days=1))


class TestQuery:
    """Test Query model."""
    
    def test_valid_query_creation(self):
        """Test creating a valid query."""
        query = Query(
            text="test search",
            entity_ids={"entity1", "entity2"},
            doc_types={DocumentType.INVOICE},
            similarity_threshold=0.5,
            max_results=20
        )
        
        assert query.text == "test search"
        assert query.entity_ids == {"entity1", "entity2"}
        assert query.doc_types == {DocumentType.INVOICE}
        assert query.similarity_threshold == 0.5
        assert query.max_results == 20
    
    def test_empty_text_validation(self):
        """Test validation for empty query text."""
        with pytest.raises(ValueError, match="Query text cannot be empty"):
            Query(text="")
    
    def test_invalid_similarity_threshold(self):
        """Test validation for invalid similarity threshold."""
        with pytest.raises(ValueError, match="Similarity threshold must be between 0.0 and 1.0"):
            Query(text="test", similarity_threshold=1.5)
        
        with pytest.raises(ValueError, match="Similarity threshold must be between 0.0 and 1.0"):
            Query(text="test", similarity_threshold=-0.1)
    
    def test_invalid_max_results(self):
        """Test validation for invalid max results."""
        with pytest.raises(ValueError, match="Max results must be positive"):
            Query(text="test", max_results=0)
        
        with pytest.raises(ValueError, match="Max results cannot exceed 1000"):
            Query(text="test", max_results=1001)


class TestSearchResult:
    """Test SearchResult model."""
    
    def test_valid_search_result(self, sample_documents):
        """Test creating a valid search result."""
        document = sample_documents[0]
        
        result = SearchResult(
            document=document,
            score=0.85,
            context_snippet="Invoice #INV-2024-001 from ACME Corp...",
            matched_terms=["invoice", "acme"],
            rank=1
        )
        
        assert result.document == document
        assert result.score == 0.85
        assert result.rank == 1
        assert "invoice" in result.matched_terms
    
    def test_invalid_score_validation(self, sample_documents):
        """Test validation for invalid score."""
        document = sample_documents[0]
        
        with pytest.raises(ValueError, match="Score must be between 0.0 and 1.0"):
            SearchResult(
                document=document,
                score=1.5,
                context_snippet="test snippet",
                matched_terms=["test"],
                rank=1
            )
    
    def test_invalid_rank_validation(self, sample_documents):
        """Test validation for invalid rank."""
        document = sample_documents[0]
        
        with pytest.raises(ValueError, match="Rank must be positive"):
            SearchResult(
                document=document,
                score=0.5,
                context_snippet="test snippet",
                matched_terms=["test"],
                rank=0
            )
    
    def test_empty_context_snippet_validation(self, sample_documents):
        """Test validation for empty context snippet."""
        document = sample_documents[0]
        
        with pytest.raises(ValueError, match="Context snippet cannot be empty"):
            SearchResult(
                document=document,
                score=0.5,
                context_snippet="",
                matched_terms=["test"],
                rank=1
            )
    
    def test_to_dict_serialization(self, sample_documents):
        """Test search result serialization to dict."""
        document = sample_documents[0]
        
        result = SearchResult(
            document=document,
            score=0.8567,
            context_snippet="Test snippet",
            matched_terms=["test", "invoice"],
            rank=1
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["score"] == 0.8567  # Should round to 4 decimal places
        assert result_dict["rank"] == 1
        assert result_dict["context_snippet"] == "Test snippet"
        assert result_dict["matched_terms"] == ["test", "invoice"]
        assert result_dict["document"]["id"] == document.id
        assert result_dict["document"]["doc_type"] == document.doc_type.value
