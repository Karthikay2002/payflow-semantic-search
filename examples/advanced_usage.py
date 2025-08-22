"""Advanced usage examples for semantic search system."""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

from semantic_search import SemanticSearchService, Document, DocumentType, Query, DateRange


async def advanced_filtering_demo():
    """Demonstrate advanced filtering capabilities."""
    print("🔍 Advanced Filtering Demo")
    print("=" * 40)
    
    async with SemanticSearchService.create(
        index_path=Path("./advanced_demo_index"),
        max_features=3000,
        log_level="WARNING"
    ) as service:
        
        # Load sample data
        sample_data_file = Path(__file__).parent / "sample_data" / "sample_documents.json"
        if not sample_data_file.exists():
            from sample_data.generate_sample_data import save_sample_documents
            save_sample_documents(sample_data_file.parent)
        
        # Load and add documents
        with open(sample_data_file) as f:
            data = json.load(f)
        
        documents = []
        for item in data:
            doc = Document(
                id=item["id"],
                content=item["content"],
                entity_id=item["entity_id"],
                doc_type=DocumentType(item["doc_type"]),
                date=datetime.fromisoformat(item["date"]),
                metadata=item["metadata"]
            )
            documents.append(doc)
        
        await service.add_documents(documents)
        print(f"Loaded {len(documents)} documents")
        
        # 1. Date range filtering
        print("\n1. Date Range Filtering:")
        
        # Recent documents (last 30 days)
        recent_date = datetime.now() - timedelta(days=30)
        recent_query = Query(
            text="invoice payment",
            date_range=DateRange(start=recent_date, end=None),
            similarity_threshold=0.1,
            max_results=5
        )
        
        recent_results = await service.search(recent_query)
        print(f"   Recent documents (last 30 days): {len(recent_results)} results")
        
        # Older documents (more than 60 days ago)
        old_date = datetime.now() - timedelta(days=60)
        old_query = Query(
            text="invoice payment",
            date_range=DateRange(start=None, end=old_date),
            similarity_threshold=0.1,
            max_results=5
        )
        
        old_results = await service.search(old_query)
        print(f"   Older documents (>60 days ago): {len(old_results)} results")
        
        # Specific date range
        start_date = datetime.now() - timedelta(days=90)
        end_date = datetime.now() - timedelta(days=30)
        range_query = Query(
            text="software services",
            date_range=DateRange(start=start_date, end=end_date),
            similarity_threshold=0.1
        )
        
        range_results = await service.search(range_query)
        print(f"   Specific range (30-90 days ago): {len(range_results)} results")
        
        # 2. Multi-entity filtering
        print("\n2. Multi-Entity Filtering:")
        
        tech_entities = {"techcorp_solutions", "digital_dynamics", "innovation_labs"}
        tech_query = Query(
            text="software development consulting",
            entity_ids=tech_entities,
            similarity_threshold=0.1
        )
        
        tech_results = await service.search(tech_query)
        print(f"   Tech companies only: {len(tech_results)} results")
        
        for result in tech_results[:3]:
            print(f"     - {result.document.entity_id}: {result.score:.3f}")
        
        # 3. Document type combinations
        print("\n3. Document Type Combinations:")
        
        financial_docs = {DocumentType.INVOICE, DocumentType.RECEIPT}
        financial_query = Query(
            text="payment amount total",
            doc_types=financial_docs,
            similarity_threshold=0.1
        )
        
        financial_results = await service.search(financial_query)
        print(f"   Financial documents (invoices + receipts): {len(financial_results)} results")
        
        # 4. Complex combined filtering
        print("\n4. Complex Combined Filtering:")
        
        complex_query = Query(
            text="software licensing subscription",
            entity_ids={"techcorp_solutions", "microsoft_corporation", "adobe_systems"},
            doc_types={DocumentType.INVOICE, DocumentType.CONTRACT},
            date_range=DateRange(
                start=datetime.now() - timedelta(days=180),
                end=datetime.now()
            ),
            similarity_threshold=0.15,
            max_results=10
        )
        
        complex_results = await service.search(complex_query)
        print(f"   Complex filter results: {len(complex_results)} results")
        
        for result in complex_results[:3]:
            print(f"     - {result.document.doc_type.value}: {result.document.entity_id}")
            print(f"       Score: {result.score:.3f}, Date: {result.document.date.strftime('%Y-%m-%d')}")
            print(f"       Amount: ${result.document.metadata.get('amount', 'N/A')}")


async def similarity_threshold_analysis():
    """Analyze the impact of different similarity thresholds."""
    print("\n🎯 Similarity Threshold Analysis")
    print("=" * 40)
    
    async with SemanticSearchService.create(
        index_path=Path("./threshold_demo_index"),
        log_level="WARNING"
    ) as service:
        
        # Load sample data
        sample_data_file = Path(__file__).parent / "sample_data" / "sample_documents.json"
        with open(sample_data_file) as f:
            data = json.load(f)
        
        documents = []
        for item in data:
            doc = Document(
                id=item["id"],
                content=item["content"],
                entity_id=item["entity_id"],
                doc_type=DocumentType(item["doc_type"]),
                date=datetime.fromisoformat(item["date"]),
                metadata=item["metadata"]
            )
            documents.append(doc)
        
        await service.add_documents(documents)
        
        # Test different thresholds
        query_text = "software development services consulting"
        thresholds = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
        
        print(f"Query: '{query_text}'")
        print("Threshold | Results | Avg Score | Min Score | Max Score")
        print("-" * 55)
        
        for threshold in thresholds:
            results = await service.search_text(
                query_text,
                similarity_threshold=threshold,
                max_results=50
            )
            
            if results:
                scores = [r.score for r in results]
                avg_score = sum(scores) / len(scores)
                min_score = min(scores)
                max_score = max(scores)
                
                print(f"{threshold:8.2f} | {len(results):7d} | {avg_score:9.3f} | {min_score:9.3f} | {max_score:9.3f}")
            else:
                print(f"{threshold:8.2f} | {0:7d} | {'N/A':>9} | {'N/A':>9} | {'N/A':>9}")


async def performance_benchmark():
    """Benchmark search performance with different configurations."""
    print("\n⚡ Performance Benchmark")
    print("=" * 40)
    
    # Test different configurations
    configs = [
        {"max_features": 1000, "name": "Small (1K features)"},
        {"max_features": 5000, "name": "Medium (5K features)"},
        {"max_features": 10000, "name": "Large (10K features)"},
    ]
    
    # Load sample data once
    sample_data_file = Path(__file__).parent / "sample_data" / "sample_documents.json"
    with open(sample_data_file) as f:
        data = json.load(f)
    
    documents = []
    for item in data:
        doc = Document(
            id=item["id"],
            content=item["content"],
            entity_id=item["entity_id"],
            doc_type=DocumentType(item["doc_type"]),
            date=datetime.fromisoformat(item["date"]),
            metadata=item["metadata"]
        )
        documents.append(doc)
    
    test_queries = [
        "software development services",
        "office supplies purchase order",
        "monthly hosting contract",
        "business travel expenses",
        "invoice payment terms"
    ]
    
    for config in configs:
        print(f"\nTesting {config['name']}:")
        
        async with SemanticSearchService.create(
            index_path=Path(f"./perf_demo_{config['max_features']}"),
            max_features=config['max_features'],
            log_level="ERROR"  # Minimal logging for benchmarks
        ) as service:
            
            # Measure indexing time
            start_time = datetime.now()
            await service.add_documents(documents)
            index_time = (datetime.now() - start_time).total_seconds()
            
            print(f"  Indexing time: {index_time:.2f}s")
            
            # Measure search times
            search_times = []
            for query in test_queries:
                start_time = datetime.now()
                results = await service.search_text(query, max_results=10)
                search_time = (datetime.now() - start_time).total_seconds()
                search_times.append(search_time)
            
            avg_search_time = sum(search_times) / len(search_times)
            print(f"  Average search time: {avg_search_time:.3f}s")
            print(f"  Search time range: {min(search_times):.3f}s - {max(search_times):.3f}s")


async def context_snippet_showcase():
    """Showcase context snippet generation quality."""
    print("\n📝 Context Snippet Showcase")
    print("=" * 40)
    
    async with SemanticSearchService.create(
        index_path=Path("./context_demo_index"),
        log_level="WARNING"
    ) as service:
        
        # Load sample data
        sample_data_file = Path(__file__).parent / "sample_data" / "sample_documents.json"
        with open(sample_data_file) as f:
            data = json.load(f)
        
        documents = []
        for item in data:
            doc = Document(
                id=item["id"],
                content=item["content"],
                entity_id=item["entity_id"],
                doc_type=DocumentType(item["doc_type"]),
                date=datetime.fromisoformat(item["date"]),
                metadata=item["metadata"]
            )
            documents.append(doc)
        
        await service.add_documents(documents)
        
        # Test queries that should generate good context snippets
        showcase_queries = [
            "ACME Corporation software development",
            "monthly hosting fees contract",
            "business travel flight expenses",
            "office supplies purchase order"
        ]
        
        for query in showcase_queries:
            print(f"\nQuery: '{query}'")
            print("-" * (len(query) + 10))
            
            results = await service.search_text(query, max_results=3)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Document: {result.document.id} (Score: {result.score:.3f})")
                print(f"   Type: {result.document.doc_type.value}")
                print(f"   Entity: {result.document.entity_id}")
                print(f"   Matched terms: {', '.join(result.matched_terms)}")
                print(f"   Context: {result.context_snippet}")


async def main():
    """Run all advanced demos."""
    print("🚀 Semantic Search - Advanced Usage Examples")
    print("=" * 50)
    
    # Generate sample data if needed
    sample_data_file = Path(__file__).parent / "sample_data" / "sample_documents.json"
    if not sample_data_file.exists():
        print("Generating sample data...")
        from sample_data.generate_sample_data import save_sample_documents
        save_sample_documents(sample_data_file.parent)
    
    # Run all demos
    await advanced_filtering_demo()
    await similarity_threshold_analysis()
    await performance_benchmark()
    await context_snippet_showcase()
    
    print("\n✅ All advanced demos completed!")
    print("\nThese examples demonstrate:")
    print("- Complex filtering with multiple criteria")
    print("- Similarity threshold optimization")
    print("- Performance characteristics")
    print("- Context snippet quality")
    print("\nFor production deployment, see the Docker configuration and README.md")


if __name__ == "__main__":
    asyncio.run(main())
