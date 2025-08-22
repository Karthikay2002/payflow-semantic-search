"""Basic usage example for semantic search system."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from semantic_search import SemanticSearchService, Document, DocumentType, Query


async def load_sample_documents(data_file: Path) -> list[Document]:
    """Load sample documents from JSON file."""
    with open(data_file) as f:
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
    
    return documents


async def basic_search_demo():
    """Demonstrate basic search functionality."""
    print("🔍 Semantic Search System - Basic Usage Demo")
    print("=" * 50)
    
    # Initialize the search service
    print("\n1. Initializing search service...")
    async with SemanticSearchService.create(
        index_path=Path("./demo_index"),
        max_features=2000,
        log_level="INFO"
    ) as service:
        
        # Load sample documents
        print("\n2. Loading sample documents...")
        sample_data_file = Path(__file__).parent / "sample_data" / "sample_documents.json"
        
        if not sample_data_file.exists():
            print("   Generating sample data...")
            from sample_data.generate_sample_data import save_sample_documents
            save_sample_documents(sample_data_file.parent)
        
        documents = await load_sample_documents(sample_data_file)
        print(f"   Loaded {len(documents)} documents")
        
        # Add documents to the search index
        print("\n3. Building search index...")
        await service.add_documents(documents)
        
        # Get service statistics
        stats = await service.get_stats()
        print(f"   Index contains {stats['engine']['total_documents']} documents")
        print(f"   Vocabulary size: {stats['engine']['vocabulary_size']} terms")
        
        # Perform basic searches
        print("\n4. Performing searches...")
        
        search_examples = [
            ("Software and licensing", "Find documents about software licensing"),
            ("ACME Corp invoices", "Find invoices from ACME Corporation"),
            ("office supplies purchase", "Find purchase orders for office supplies"),
            ("monthly hosting fees", "Find contracts with monthly hosting fees"),
            ("business travel expenses", "Find travel-related receipts")
        ]
        
        for query_text, description in search_examples:
            print(f"\n   Query: '{query_text}' ({description})")
            
            results = await service.search_text(
                query_text,
                similarity_threshold=0.1,
                max_results=3
            )
            
            if results:
                print(f"   Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"     {i}. {result.document.doc_type.value.title()} "
                          f"({result.document.entity_id}) - Score: {result.score:.3f}")
                    print(f"        Context: {result.context_snippet[:100]}...")
                    print(f"        Matched terms: {', '.join(result.matched_terms[:5])}")
            else:
                print("   No results found")
        
        print("\n5. Advanced filtering examples...")
        
        # Example 1: Filter by document type
        print("\n   Filter by document type (invoices only):")
        results = await service.search_text(
            "software development services",
            doc_types=["invoice"],
            max_results=5
        )
        print(f"   Found {len(results)} invoice results")
        
        # Example 2: Filter by entity
        print("\n   Filter by entity (ACME Corp only):")
        results = await service.search_text(
            "development services",
            entity_ids=["acme_corporation"],
            max_results=5
        )
        print(f"   Found {len(results)} results from ACME Corp")
        
        # Example 3: Complex query with multiple filters
        print("\n   Complex query (invoices from tech companies about software):")
        results = await service.search_text(
            "software consulting development",
            entity_ids=["techcorp_solutions", "digital_dynamics", "innovation_labs"],
            doc_types=["invoice"],
            similarity_threshold=0.2,
            max_results=5
        )
        print(f"   Found {len(results)} matching results")
        
        for result in results[:2]:  # Show top 2 results
            print(f"     - {result.document.id}: Score {result.score:.3f}")
            print(f"       Entity: {result.document.entity_id}")
            print(f"       Amount: ${result.document.metadata.get('amount', 'N/A')}")
        
        print("\n6. Performance and health check...")
        
        # Health check
        health = await service.health_check()
        print(f"   System status: {health['status']}")
        print(f"   Index ready: {health['is_ready']}")
        
        # Final statistics
        final_stats = await service.get_stats()
        print(f"   Total searches performed: {final_stats['engine']['total_searches']}")
        print(f"   Average search time: {final_stats['engine']['avg_search_time']:.3f}s")
    
    print("\n✅ Demo completed successfully!")
    print("\nNext steps:")
    print("- Check out advanced_usage.py for more complex examples")
    print("- Explore the test suite to understand all features")
    print("- Review the README.md for deployment instructions")


if __name__ == "__main__":
    asyncio.run(basic_search_demo())
