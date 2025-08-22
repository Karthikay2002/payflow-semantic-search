"""
Performance benchmark for semantic search system.

Measures and compares performance characteristics of TF-IDF vs Sentence-BERT
across different document sizes and query types.
"""

import asyncio
import time
import statistics
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from semantic_search.core.embeddings import TFIDFEmbedding
from semantic_search.core.hybrid_engine import HybridSearchEngine
from semantic_search.models.document import Document, DocumentType
from semantic_search.models.query import Query

try:
    from semantic_search.core.transformer_embeddings import TRANSFORMERS_AVAILABLE
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def load_sample_documents() -> List[Document]:
    """Load sample documents for benchmarking."""
    sample_file = Path(__file__).parent / "sample_data" / "sample_documents.json"
    
    if not sample_file.exists():
        print("❌ Sample data not found. Run: python examples/sample_data/generate_sample_data.py")
        return []
    
    with open(sample_file) as f:
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


async def benchmark_indexing(documents: List[Document]) -> Dict[str, Any]:
    """Benchmark document indexing performance."""
    print("📚 Benchmarking Indexing Performance...")
    
    results = {}
    
    # TF-IDF indexing
    print("  🔤 Testing TF-IDF indexing...")
    tfidf_engine = TFIDFEmbedding()
    
    start_time = time.time()
    await tfidf_engine.add_documents(documents)
    tfidf_time = time.time() - start_time
    
    results['tfidf'] = {
        'indexing_time': tfidf_time,
        'docs_per_second': len(documents) / tfidf_time,
        'vocabulary_size': tfidf_engine.get_stats()['vocabulary_size']
    }
    
    print(f"    ⏱️  Time: {tfidf_time:.3f}s ({len(documents) / tfidf_time:.1f} docs/sec)")
    
    # Transformer indexing (if available)
    if TRANSFORMERS_AVAILABLE:
        try:
            print("  🤖 Testing Sentence-BERT indexing...")
            from semantic_search.core.transformer_embeddings import TransformerEmbedding
            
            transformer_engine = TransformerEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                use_financial_model=False
            )
            
            start_time = time.time()
            await transformer_engine.add_documents(documents)
            transformer_time = time.time() - start_time
            
            results['transformer'] = {
                'indexing_time': transformer_time,
                'docs_per_second': len(documents) / transformer_time,
                'vector_dimension': transformer_engine.get_stats()['vector_dimension']
            }
            
            print(f"    ⏱️  Time: {transformer_time:.3f}s ({len(documents) / transformer_time:.1f} docs/sec)")
            print(f"    📊 Speedup: TF-IDF is {transformer_time / tfidf_time:.1f}x faster for indexing")
            
        except Exception as e:
            print(f"    ❌ Transformer indexing failed: {e}")
            results['transformer'] = {'error': str(e)}
    else:
        print("  ⚠️  Transformers not available")
        results['transformer'] = {'error': 'Not available'}
    
    return results


async def benchmark_search(documents: List[Document]) -> Dict[str, Any]:
    """Benchmark search performance across different query types."""
    print("\n🔍 Benchmarking Search Performance...")
    
    # Test queries of different complexity
    test_queries = [
        "software",                           # Single term
        "cloud computing",                    # Two terms
        "software development consulting",    # Multiple terms
        "business travel expenses receipts"   # Complex query
    ]
    
    results = {}
    
    # Setup engines
    tfidf_engine = TFIDFEmbedding()
    await tfidf_engine.add_documents(documents)
    
    transformer_engine = None
    if TRANSFORMERS_AVAILABLE:
        try:
            from semantic_search.core.transformer_embeddings import TransformerEmbedding
            transformer_engine = TransformerEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                use_financial_model=False
            )
            await transformer_engine.add_documents(documents)
        except Exception as e:
            print(f"  ⚠️  Transformer setup failed: {e}")
    
    # Benchmark each query
    for query_text in test_queries:
        print(f"\n  🎯 Query: '{query_text}'")
        query_results = {}
        
        # TF-IDF search timing
        tfidf_times = []
        for _ in range(5):  # Run 5 times for average
            start_time = time.time()
            tfidf_results = await tfidf_engine.search(query_text, top_k=10, threshold=0.01)
            tfidf_times.append(time.time() - start_time)
        
        query_results['tfidf'] = {
            'avg_time': statistics.mean(tfidf_times),
            'min_time': min(tfidf_times),
            'max_time': max(tfidf_times),
            'results_count': len(tfidf_results)
        }
        
        print(f"    🔤 TF-IDF: {statistics.mean(tfidf_times):.3f}s avg, {len(tfidf_results)} results")
        
        # Transformer search timing
        if transformer_engine:
            try:
                transformer_times = []
                for _ in range(3):  # Fewer runs since it's slower
                    start_time = time.time()
                    transformer_results = await transformer_engine.search(query_text, top_k=10, threshold=0.01)
                    transformer_times.append(time.time() - start_time)
                
                query_results['transformer'] = {
                    'avg_time': statistics.mean(transformer_times),
                    'min_time': min(transformer_times),
                    'max_time': max(transformer_times),
                    'results_count': len(transformer_results)
                }
                
                avg_transformer_time = statistics.mean(transformer_times)
                avg_tfidf_time = statistics.mean(tfidf_times)
                speedup = avg_transformer_time / avg_tfidf_time
                
                print(f"    🤖 Transformer: {avg_transformer_time:.3f}s avg, {len(transformer_results)} results")
                print(f"    📊 TF-IDF is {speedup:.1f}x faster")
                
            except Exception as e:
                print(f"    ❌ Transformer search failed: {e}")
                query_results['transformer'] = {'error': str(e)}
        
        results[query_text] = query_results
    
    return results


async def benchmark_scaling(documents: List[Document]) -> Dict[str, Any]:
    """Benchmark performance scaling with document count."""
    print("\n📈 Benchmarking Scaling Performance...")
    
    results = {}
    doc_counts = [10, 25, 50, len(documents)]
    
    for doc_count in doc_counts:
        if doc_count > len(documents):
            continue
            
        print(f"\n  📊 Testing with {doc_count} documents...")
        subset_docs = documents[:doc_count]
        
        # TF-IDF scaling
        tfidf_engine = TFIDFEmbedding()
        
        start_time = time.time()
        await tfidf_engine.add_documents(subset_docs)
        indexing_time = time.time() - start_time
        
        start_time = time.time()
        search_results = await tfidf_engine.search("software services", top_k=5, threshold=0.01)
        search_time = time.time() - start_time
        
        results[doc_count] = {
            'tfidf': {
                'indexing_time': indexing_time,
                'search_time': search_time,
                'results_count': len(search_results),
                'vocabulary_size': tfidf_engine.get_stats()['vocabulary_size']
            }
        }
        
        print(f"    🔤 TF-IDF: Index {indexing_time:.3f}s, Search {search_time:.3f}s")
    
    return results


async def run_comprehensive_benchmark():
    """Run comprehensive performance benchmark."""
    print("🚀 Financial Semantic Search - Performance Benchmark")
    print("=" * 60)
    
    # Load documents
    documents = load_sample_documents()
    if not documents:
        return
    
    print(f"📄 Loaded {len(documents)} financial documents for benchmarking")
    
    # Run benchmarks
    indexing_results = await benchmark_indexing(documents)
    search_results = await benchmark_search(documents)
    scaling_results = await benchmark_scaling(documents)
    
    # Generate report
    print("\n📊 BENCHMARK REPORT")
    print("=" * 60)
    
    # Indexing performance
    print("\n🏗️  Indexing Performance:")
    for method, data in indexing_results.items():
        if 'error' not in data:
            print(f"  {method.upper():<12}: {data['indexing_time']:.3f}s ({data['docs_per_second']:.1f} docs/sec)")
        else:
            print(f"  {method.upper():<12}: {data['error']}")
    
    # Search performance
    print("\n🔍 Search Performance (Average Times):")
    for query, query_data in search_results.items():
        print(f"\n  Query: '{query}'")
        for method, data in query_data.items():
            if 'error' not in data:
                print(f"    {method.upper():<12}: {data['avg_time']:.3f}s ({data['results_count']} results)")
            else:
                print(f"    {method.upper():<12}: {data['error']}")
    
    # Scaling analysis
    print("\n📈 Scaling Performance:")
    for doc_count, scaling_data in scaling_results.items():
        tfidf_data = scaling_data['tfidf']
        print(f"  {doc_count:>3} docs: Index {tfidf_data['indexing_time']:.3f}s, Search {tfidf_data['search_time']:.3f}s")
    
    # Save results
    benchmark_results = {
        'timestamp': datetime.now().isoformat(),
        'document_count': len(documents),
        'indexing': indexing_results,
        'search': search_results,
        'scaling': scaling_results
    }
    
    results_file = Path(__file__).parent / "benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump(benchmark_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("\n✅ Benchmark completed!")


async def main():
    """Main benchmark entry point."""
    try:
        await run_comprehensive_benchmark()
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
