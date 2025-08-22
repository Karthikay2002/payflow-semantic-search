"""Performance tests for semantic search system."""

import pytest
import asyncio
import time
from datetime import datetime, timedelta

from semantic_search.models.document import Document, DocumentType
from semantic_search.api.service import SemanticSearchService


class TestPerformance:
    """Performance and load testing."""
    
    @pytest.fixture
    def large_document_set(self):
        """Create a large set of documents for performance testing."""
        documents = []
        base_date = datetime.now()
        
        # Create diverse document types and content
        content_templates = [
            "Invoice #{doc_id} from {entity}. Total amount: ${amount}. Services: {service}. Due date: {date}.",
            "Purchase Order #{doc_id} for {entity}. Items: {items}. Total: ${amount}. Delivery: {date}.",
            "Contract with {entity}. Monthly fee: ${amount}. Terms: {terms}. Effective: {date}.",
            "Receipt from {entity}. Amount: ${amount}. Category: {category}. Date: {date}.",
            "Statement for {entity}. Balance: ${amount}. Period: {period}. Due: {date}."
        ]
        
        entities = ["ACME Corp", "TechCorp", "Office Depot", "CloudHost", "Global Services"]
        services = ["consulting", "software development", "office supplies", "hosting", "support"]
        
        for i in range(1000):  # Create 1000 documents
            template = content_templates[i % len(content_templates)]
            entity = entities[i % len(entities)]
            
            content = template.format(
                doc_id=f"DOC-{i:04d}",
                entity=entity,
                amount=round(100 + (i * 37.5) % 10000, 2),
                service=services[i % len(services)],
                items=f"Item {i % 10}, Item {(i+1) % 10}",
                terms=f"{12 + (i % 24)} months",
                category="business",
                period="monthly",
                date=(base_date - timedelta(days=i % 365)).strftime("%Y-%m-%d")
            )
            
            doc = Document(
                id=f"perf_doc_{i:04d}",
                content=content,
                entity_id=entity.lower().replace(" ", "_"),
                doc_type=DocumentType(list(DocumentType)[i % len(DocumentType)]),
                date=base_date - timedelta(days=i % 365),
                metadata={"performance_test": True, "batch": i // 100}
            )
            documents.append(doc)
        
        return documents
    
    @pytest.mark.asyncio
    async def test_large_index_creation_performance(self, temp_index_path, large_document_set):
        """Test performance of creating large index."""
        start_time = time.time()
        
        async with SemanticSearchService.create(
            index_path=temp_index_path,
            max_features=5000,
            max_workers=4
        ) as service:
            
            # Add documents in batches
            batch_size = 100
            for i in range(0, len(large_document_set), batch_size):
                batch = large_document_set[i:i + batch_size]
                await service.add_documents(batch)
            
            indexing_time = time.time() - start_time
            
            # Verify all documents are indexed
            stats = await service.get_stats()
            assert stats['engine']['total_documents'] == len(large_document_set)
            
            print(f"Indexed {len(large_document_set)} documents in {indexing_time:.2f}s")
            print(f"Average: {indexing_time / len(large_document_set) * 1000:.2f}ms per document")
            
            # Performance assertions (adjust based on expected performance)
            assert indexing_time < 60.0  # Should complete within 60 seconds
            assert indexing_time / len(large_document_set) < 0.1  # Less than 100ms per document
    
    @pytest.mark.asyncio
    async def test_search_performance(self, temp_index_path, large_document_set):
        """Test search performance on large index."""
        async with SemanticSearchService.create(
            index_path=temp_index_path,
            max_features=5000
        ) as service:
            
            # Create index
            await service.add_documents(large_document_set)
            
            # Test various search scenarios
            search_queries = [
                "invoice ACME Corp software development",
                "purchase order office supplies",
                "contract hosting monthly fee",
                "receipt business expenses",
                "statement balance payment due"
            ]
            
            search_times = []
            
            for query in search_queries:
                start_time = time.time()
                results = await service.search_text(
                    query,
                    similarity_threshold=0.1,
                    max_results=50
                )
                search_time = time.time() - start_time
                search_times.append(search_time)
                
                # Verify results quality
                assert len(results) > 0
                assert len(results) <= 50
                
                print(f"Query '{query}' returned {len(results)} results in {search_time:.3f}s")
            
            avg_search_time = sum(search_times) / len(search_times)
            print(f"Average search time: {avg_search_time:.3f}s")
            
            # Performance assertions
            assert avg_search_time < 1.0  # Average search should be under 1 second
            assert max(search_times) < 2.0  # No single search should take more than 2 seconds
    
    @pytest.mark.asyncio
    async def test_concurrent_search_performance(self, temp_index_path, large_document_set):
        """Test performance under concurrent search load."""
        async with SemanticSearchService.create(
            index_path=temp_index_path,
            max_features=5000,
            max_workers=4
        ) as service:
            
            # Create index
            await service.add_documents(large_document_set[:500])  # Smaller set for faster setup
            
            # Prepare concurrent queries
            queries = [
                "invoice software development",
                "purchase order supplies",
                "contract hosting",
                "receipt expenses",
                "statement payment"
            ] * 10  # 50 total queries
            
            # Execute concurrent searches
            start_time = time.time()
            
            tasks = [
                service.search_text(query, similarity_threshold=0.1, max_results=20)
                for query in queries
            ]
            
            results_list = await asyncio.gather(*tasks)
            
            total_time = time.time() - start_time
            
            # Verify all searches completed successfully
            assert len(results_list) == len(queries)
            for results in results_list:
                assert isinstance(results, list)
                assert len(results) <= 20
            
            print(f"Completed {len(queries)} concurrent searches in {total_time:.2f}s")
            print(f"Average time per search: {total_time / len(queries):.3f}s")
            
            # Performance assertions
            assert total_time < 30.0  # All searches should complete within 30 seconds
            assert total_time / len(queries) < 1.0  # Average should be under 1 second per search
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, temp_index_path, large_document_set):
        """Test memory usage remains stable during operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        async with SemanticSearchService.create(
            index_path=temp_index_path,
            max_features=3000
        ) as service:
            
            # Add documents in batches and monitor memory
            batch_size = 100
            memory_readings = [initial_memory]
            
            for i in range(0, min(500, len(large_document_set)), batch_size):
                batch = large_document_set[i:i + batch_size]
                await service.add_documents(batch)
                
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_readings.append(current_memory)
            
            # Perform searches and monitor memory
            for _ in range(20):
                await service.search_text("test query", similarity_threshold=0.1)
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_readings.append(current_memory)
            
            final_memory = memory_readings[-1]
            peak_memory = max(memory_readings)
            
            print(f"Initial memory: {initial_memory:.1f} MB")
            print(f"Peak memory: {peak_memory:.1f} MB")
            print(f"Final memory: {final_memory:.1f} MB")
            print(f"Memory increase: {final_memory - initial_memory:.1f} MB")
            
            # Memory should not grow excessively
            memory_growth = final_memory - initial_memory
            assert memory_growth < 500  # Should not use more than 500MB additional
            
            # Peak memory should be reasonable
            assert peak_memory < initial_memory + 1000  # Peak should not exceed 1GB additional
    
    @pytest.mark.asyncio
    async def test_index_persistence_performance(self, temp_index_path, large_document_set):
        """Test performance of index saving and loading."""
        # Create and populate service
        async with SemanticSearchService.create(
            index_path=temp_index_path,
            max_features=3000
        ) as service:
            
            await service.add_documents(large_document_set[:500])
            
            # Test save performance
            start_time = time.time()
            await service.save_index()
            save_time = time.time() - start_time
            
            print(f"Index save time: {save_time:.2f}s")
            assert save_time < 10.0  # Should save within 10 seconds
        
        # Test load performance
        start_time = time.time()
        async with SemanticSearchService.create(
            index_path=temp_index_path,
            load_existing=True
        ) as service:
            
            load_time = time.time() - start_time
            
            # Verify index loaded correctly
            stats = await service.get_stats()
            assert stats['engine']['total_documents'] == 500
            
            # Test search on loaded index
            results = await service.search_text("test query")
            
            print(f"Index load time: {load_time:.2f}s")
            assert load_time < 10.0  # Should load within 10 seconds
