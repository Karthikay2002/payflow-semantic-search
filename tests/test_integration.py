"""Integration tests for the complete semantic search system."""

import pytest
import asyncio
from datetime import datetime, timedelta

from semantic_search.api.service import SemanticSearchService
from semantic_search.models.document import Document, DocumentType
from semantic_search.models.query import Query, DateRange
from semantic_search.core.exceptions import SemanticSearchError


class TestSemanticSearchServiceIntegration:
    """Integration tests for the complete service."""
    
    async def test_full_workflow(self, temp_index_path, sample_documents):
        """Test complete workflow from service creation to search."""
        async with SemanticSearchService.create(
            index_path=temp_index_path,
            max_features=1000,
            log_level="WARNING"
        ) as service:
            
            # Add documents
            await service.add_documents(sample_documents)
            
            # Perform various searches
            results1 = await service.search_text("invoice ACME")
            assert len(results1) > 0
            
            results2 = await service.search_text(
                "software licensing",
                entity_ids=["techcorp"]
            )
            assert len(results2) > 0
            assert all(r.document.entity_id == "techcorp" for r in results2)
            
            # Test with filters
            results3 = await service.search_text(
                "payment",
                doc_types=["invoice"],
                max_results=2
            )
            assert len(results3) <= 2
    
    async def test_service_persistence(self, temp_index_path, sample_documents):
        """Test service persistence across sessions."""
        # First session: create and populate
        async with SemanticSearchService.create(
            index_path=temp_index_path,
            auto_save=True
        ) as service1:
            await service1.add_documents(sample_documents)
            
            # Verify documents are added
            results = await service1.search_text("invoice")
            assert len(results) > 0
        
        # Second session: load existing data
        async with SemanticSearchService.create(
            index_path=temp_index_path,
            load_existing=True
        ) as service2:
            # Should be able to search without adding documents again
            results = await service2.search_text("invoice")
            assert len(results) > 0
    
    async def test_concurrent_searches(self, populated_service):
        """Test concurrent search operations."""
        # Create multiple search queries
        queries = [
            "invoice ACME",
            "software licensing",
            "office supplies",
            "hosting services",
            "travel expenses"
        ]
        
        # Execute searches concurrently
        tasks = [
            populated_service.search_text(query, similarity_threshold=0.1)
            for query in queries
        ]
        
        results_list = await asyncio.gather(*tasks)
        
        # All searches should complete successfully
        assert len(results_list) == len(queries)
        for results in results_list:
            assert isinstance(results, list)
    
    async def test_large_document_batch(self, temp_index_path):
        """Test handling of large document batches."""
        # Create a large batch of documents
        large_batch = []
        base_date = datetime.now()
        
        for i in range(100):
            doc = Document(
                id=f"doc_{i:03d}",
                content=f"Document {i} with unique content about topic {i % 10}. "
                       f"This document contains various financial terms and amounts.",
                entity_id=f"entity_{i % 5}",
                doc_type=DocumentType.INVOICE if i % 2 == 0 else DocumentType.PURCHASE_ORDER,
                date=base_date - timedelta(days=i),
                metadata={"batch_id": i // 10, "amount": 100.0 + i}
            )
            large_batch.append(doc)
        
        async with SemanticSearchService.create(
            index_path=temp_index_path,
            max_features=2000
        ) as service:
            
            # Add large batch
            await service.add_documents(large_batch)
            
            # Verify all documents are indexed
            stats = await service.get_stats()
            assert stats['engine']['total_documents'] == 100
            
            # Test search performance
            results = await service.search_text("financial terms", max_results=20)
            assert len(results) > 0
    
    async def test_error_handling(self, temp_index_path):
        """Test comprehensive error handling."""
        async with SemanticSearchService.create(index_path=temp_index_path) as service:
            
            # Test invalid document
            invalid_doc = Document(
                id="",  # Invalid empty ID
                content="Test content",
                entity_id="test_entity",
                doc_type=DocumentType.INVOICE,
                date=datetime.now()
            )
            
            with pytest.raises(SemanticSearchError):
                await service.add_document(invalid_doc)
            
            # Test search on empty index
            with pytest.raises(SemanticSearchError):
                await service.search_text("test query")
    
    async def test_health_monitoring(self, populated_service):
        """Test health check and statistics monitoring."""
        # Check initial health
        health = await populated_service.health_check()
        assert health['status'] == 'healthy'
        assert health['is_ready']
        
        # Check statistics
        stats = await populated_service.get_stats()
        assert stats['service']['initialized']
        assert stats['engine']['total_documents'] > 0
        
        # Perform searches and check updated stats
        await populated_service.search_text("test query")
        
        updated_stats = await populated_service.get_stats()
        assert updated_stats['engine']['total_searches'] > 0
    
    async def test_complex_query_scenarios(self, populated_service):
        """Test complex query scenarios with multiple filters."""
        # Test 1: Entity and document type filters
        results = await populated_service.search_text(
            "payment invoice",
            entity_ids=["acme_corp", "techcorp"],
            doc_types=["invoice"]
        )
        
        for result in results:
            assert result.document.entity_id in ["acme_corp", "techcorp"]
            assert result.document.doc_type == DocumentType.INVOICE
        
        # Test 2: Date range filtering
        recent_date = datetime.now() - timedelta(days=20)
        query = Query(
            text="invoice",
            date_range=DateRange(start=recent_date, end=None),
            similarity_threshold=0.1
        )
        
        results = await populated_service.search(query)
        for result in results:
            assert result.document.date >= recent_date
        
        # Test 3: High similarity threshold
        results = await populated_service.search_text(
            "very specific unique terms that probably dont exist",
            similarity_threshold=0.9
        )
        # Should return few or no results due to high threshold
        assert len(results) <= 1
    
    async def test_context_and_relevance(self, populated_service):
        """Test context snippet generation and relevance scoring."""
        results = await populated_service.search_text(
            "ACME Corp invoice development",
            similarity_threshold=0.1
        )
        
        assert len(results) > 0
        
        for result in results:
            # Context snippet should be shorter than full content
            assert len(result.context_snippet) < len(result.document.content)
            
            # Should contain some query terms
            snippet_lower = result.context_snippet.lower()
            assert any(term in snippet_lower for term in ["acme", "invoice", "development"])
            
            # Matched terms should be relevant
            assert len(result.matched_terms) > 0
            
            # Score should be reasonable
            assert 0.0 <= result.score <= 1.0
            
            # Rank should be positive
            assert result.rank > 0
        
        # Results should be ranked by relevance (descending score)
        scores = [result.score for result in results]
        assert scores == sorted(scores, reverse=True)
    
    async def test_service_lifecycle(self, temp_index_path, sample_documents):
        """Test complete service lifecycle management."""
        service = SemanticSearchService(
            index_path=temp_index_path,
            auto_save=True,
            log_level="WARNING"
        )
        
        # Test initialization
        await service.initialize(load_existing=False)
        assert service._initialized
        
        # Test operations
        await service.add_documents(sample_documents)
        results = await service.search_text("invoice")
        assert len(results) > 0
        
        # Test cleanup
        await service.close()
        assert not service._initialized
