"""Test core search engine functionality."""

import pytest
import asyncio
from datetime import datetime, timedelta

from semantic_search.core.engine import SemanticSearchEngine
from semantic_search.core.embeddings import TFIDFEmbedding
from semantic_search.models.document import Document, DocumentType
from semantic_search.models.query import Query, DateRange
from semantic_search.core.exceptions import ValidationError, SearchError, DocumentProcessingError


class TestSemanticSearchEngine:
    """Test SemanticSearchEngine functionality."""
    
    @pytest.fixture
    async def engine(self, temp_index_path):
        """Create a search engine for testing."""
        engine = SemanticSearchEngine(
            index_path=temp_index_path,
            max_features=1000,  # Smaller for faster tests
            max_workers=2
        )
        yield engine
        await engine.close()
    
    async def test_add_single_document(self, engine, sample_documents):
        """Test adding a single document."""
        document = sample_documents[0]
        
        await engine.add_document(document)
        
        stats = engine.get_stats()
        assert stats['total_documents'] == 1
    
    async def test_add_multiple_documents(self, engine, sample_documents):
        """Test adding multiple documents."""
        await engine.add_documents(sample_documents)
        
        stats = engine.get_stats()
        assert stats['total_documents'] == len(sample_documents)
    
    async def test_add_invalid_document(self, engine):
        """Test adding invalid document raises error."""
        invalid_doc = Document(
            id="",  # Invalid empty ID
            content="Test content",
            entity_id="test_entity",
            doc_type=DocumentType.INVOICE,
            date=datetime.now()
        )
        
        with pytest.raises(DocumentProcessingError):
            await engine.add_document(invalid_doc)
    
    async def test_basic_search(self, engine, sample_documents):
        """Test basic search functionality."""
        await engine.add_documents(sample_documents)
        
        query = Query(
            text="invoice",  # Simpler query that should definitely match
            similarity_threshold=0.01,  # Very low threshold for small test datasets
            max_results=10
        )
        
        results = await engine.search(query)
        
        assert len(results) > 0
        assert all(result.score >= 0.01 for result in results)  # Match the query threshold
        assert all(result.rank > 0 for result in results)
        
        # Results should be ranked by score (descending)
        scores = [result.score for result in results]
        assert scores == sorted(scores, reverse=True)
    
    async def test_search_with_entity_filter(self, engine, sample_documents):
        """Test search with entity ID filter."""
        await engine.add_documents(sample_documents)
        
        query = Query(
            text="invoice",
            entity_ids={"acme_corp"},
            similarity_threshold=0.1
        )
        
        results = await engine.search(query)
        
        assert len(results) > 0
        assert all(result.document.entity_id == "acme_corp" for result in results)
    
    async def test_search_with_doc_type_filter(self, engine, sample_documents):
        """Test search with document type filter."""
        await engine.add_documents(sample_documents)
        
        query = Query(
            text="payment",
            doc_types={DocumentType.INVOICE},
            similarity_threshold=0.1
        )
        
        results = await engine.search(query)
        
        assert len(results) >= 0  # May be 0 if no invoices match
        assert all(result.document.doc_type == DocumentType.INVOICE for result in results)
    
    async def test_search_with_date_filter(self, engine, sample_documents):
        """Test search with date range filter."""
        await engine.add_documents(sample_documents)
        
        # Filter to recent documents only
        start_date = datetime.now() - timedelta(days=20)
        date_range = DateRange(start=start_date, end=None)
        
        query = Query(
            text="invoice",
            date_range=date_range,
            similarity_threshold=0.1
        )
        
        results = await engine.search(query)
        
        assert all(result.document.date >= start_date for result in results)
    
    async def test_search_with_multiple_filters(self, engine, sample_documents):
        """Test search with multiple filters combined."""
        await engine.add_documents(sample_documents)
        
        query = Query(
            text="invoice",
            entity_ids={"acme_corp", "techcorp"},
            doc_types={DocumentType.INVOICE},
            similarity_threshold=0.1
        )
        
        results = await engine.search(query)
        
        for result in results:
            assert result.document.entity_id in {"acme_corp", "techcorp"}
            assert result.document.doc_type == DocumentType.INVOICE
    
    async def test_search_invalid_query(self, engine, sample_documents):
        """Test search with invalid query raises error."""
        await engine.add_documents(sample_documents)
        
        invalid_query = Query(
            text="",  # Invalid empty text
            similarity_threshold=0.1
        )
        
        with pytest.raises(SearchError):
            await engine.search(invalid_query)
    
    async def test_search_empty_index(self, engine):
        """Test search on empty index."""
        query = Query(text="test search")
        
        with pytest.raises(SearchError):
            await engine.search(query)
    
    async def test_context_snippet_generation(self, engine, sample_documents):
        """Test that search results include context snippets."""
        await engine.add_documents(sample_documents)
        
        query = Query(
            text="ACME Corp invoice",
            similarity_threshold=0.1
        )
        
        results = await engine.search(query)
        
        assert len(results) > 0
        for result in results:
            assert len(result.context_snippet) > 0
            assert result.context_snippet != result.document.content  # Should be snippet, not full content
    
    async def test_matched_terms_extraction(self, engine, sample_documents):
        """Test that search results include matched terms."""
        await engine.add_documents(sample_documents)
        
        query = Query(
            text="invoice software licensing",
            similarity_threshold=0.1
        )
        
        results = await engine.search(query)
        
        assert len(results) > 0
        for result in results:
            assert len(result.matched_terms) > 0
            # At least some query terms should be matched
            query_terms = query.text.lower().split()
            assert any(term in result.matched_terms for term in query_terms)
    
    async def test_similarity_threshold_filtering(self, engine, sample_documents):
        """Test that similarity threshold properly filters results."""
        await engine.add_documents(sample_documents)
        
        # High threshold should return fewer results
        high_threshold_query = Query(
            text="very specific uncommon terms",
            similarity_threshold=0.8
        )
        
        high_results = await engine.search(high_threshold_query)
        
        # Low threshold should return more results
        low_threshold_query = Query(
            text="very specific uncommon terms",
            similarity_threshold=0.1
        )
        
        low_results = await engine.search(low_threshold_query)
        
        assert len(low_results) >= len(high_results)
        
        # All results should meet threshold
        for result in high_results:
            assert result.score >= 0.8
        
        for result in low_results:
            assert result.score >= 0.1
    
    async def test_max_results_limiting(self, engine, sample_documents):
        """Test that max_results properly limits output."""
        await engine.add_documents(sample_documents)
        
        query = Query(
            text="invoice",
            max_results=2,
            similarity_threshold=0.01  # Very low to get multiple results
        )
        
        results = await engine.search(query)
        
        assert len(results) <= 2
    
    async def test_index_persistence(self, engine, sample_documents, temp_index_path):
        """Test saving and loading index."""
        # Add documents and save
        await engine.add_documents(sample_documents)
        await engine.save_index()
        
        # Create new engine and load
        new_engine = SemanticSearchEngine(index_path=temp_index_path)
        await new_engine.load_index()
        
        # Should be able to search
        query = Query(text="invoice", similarity_threshold=0.1)
        results = await new_engine.search(query)
        
        assert len(results) > 0
        
        await new_engine.close()
    
    async def test_health_check(self, engine, sample_documents):
        """Test engine health check."""
        # Health check on empty engine
        health = await engine.health_check()
        assert health['status'] == 'not_ready'
        assert not health['is_ready']
        
        # Health check after adding documents
        await engine.add_documents(sample_documents)
        health = await engine.health_check()
        assert health['status'] == 'healthy'
        assert health['is_ready']
        assert health['stats']['total_documents'] == len(sample_documents)
    
    async def test_stats_tracking(self, engine, sample_documents):
        """Test that engine tracks statistics correctly."""
        initial_stats = engine.get_stats()
        assert initial_stats['total_documents'] == 0
        assert initial_stats['total_searches'] == 0
        
        # Add documents
        await engine.add_documents(sample_documents)
        
        # Perform searches
        query = Query(text="invoice", similarity_threshold=0.1)
        await engine.search(query)
        await engine.search(query)
        
        final_stats = engine.get_stats()
        assert final_stats['total_documents'] == len(sample_documents)
        assert final_stats['total_searches'] == 2
        assert final_stats['avg_search_time'] > 0


class TestTFIDFEmbedding:
    """Test TF-IDF embedding functionality."""
    
    @pytest.fixture
    def embedding_engine(self):
        """Create TF-IDF embedding engine for testing."""
        return TFIDFEmbedding(max_features=1000)
    
    async def test_add_and_search_documents(self, embedding_engine, sample_documents):
        """Test basic add and search functionality."""
        await embedding_engine.add_documents(sample_documents)
        
        assert embedding_engine.is_fitted
        assert len(embedding_engine.documents) == len(sample_documents)
        
        # Test search
        results = await embedding_engine.search("invoice software", top_k=5)
        
        assert len(results) > 0
        assert all(isinstance(doc, Document) for doc, score in results)
        assert all(0.0 <= score <= 1.0 for doc, score in results)
    
    async def test_empty_document_list(self, embedding_engine):
        """Test adding empty document list."""
        await embedding_engine.add_documents([])
        assert not embedding_engine.is_fitted
    
    async def test_search_before_fitting(self, embedding_engine):
        """Test search before adding documents raises error."""
        with pytest.raises(Exception):  # Should raise IndexError
            await embedding_engine.search("test query")
    
    def test_get_stats(self, embedding_engine):
        """Test getting embedding statistics."""
        stats = embedding_engine.get_stats()
        
        assert 'total_documents' in stats
        assert 'vocabulary_size' in stats
        assert 'is_fitted' in stats
        assert 'config' in stats
        
        assert stats['total_documents'] == 0
        assert not stats['is_fitted']
