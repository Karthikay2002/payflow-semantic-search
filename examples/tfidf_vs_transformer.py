"""
Simple comparison of TF-IDF vs Sentence-BERT on financial documents.

This demonstrates the core comparison they requested:
- TF-IDF as baseline (fast, keyword-based)
- Sentence-BERT as SOTA (semantic understanding)
- Side-by-side results on real financial data
"""

import asyncio
import logging
import json
from pathlib import Path
from datetime import datetime

from semantic_search.core.embeddings import TFIDFEmbedding
from semantic_search.models.document import Document, DocumentType
from semantic_search.models.query import Query

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise


def load_sample_documents() -> list[Document]:
    """Load the existing sample financial documents."""
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


async def compare_search_methods():
    """Compare TF-IDF vs Sentence-BERT on financial documents."""
    print("🔍 Financial Semantic Search - TF-IDF vs Sentence-BERT Comparison")
    print("=" * 70)
    
    # Load documents
    print("\n📄 Loading sample financial documents...")
    documents = load_sample_documents()
    
    if not documents:
        return
    
    print(f"✅ Loaded {len(documents)} documents")
    
    # Initialize TF-IDF engine
    print("\n🔧 Initializing TF-IDF search engine...")
    tfidf_engine = TFIDFEmbedding()
    await tfidf_engine.add_documents(documents)
    print(f"✅ TF-IDF index built - vocabulary size: {tfidf_engine.get_stats()['vocabulary_size']}")
    
    # Try to initialize Sentence-BERT
    print("\n🤖 Attempting to initialize Sentence-BERT...")
    transformer_engine = None
    
    try:
        from semantic_search.core.transformer_embeddings import TransformerEmbedding, TRANSFORMERS_AVAILABLE
        
        if TRANSFORMERS_AVAILABLE:
            transformer_engine = TransformerEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                use_financial_model=False  # Simplified
            )
            await transformer_engine.add_documents(documents)
            print(f"✅ Sentence-BERT ready - {transformer_engine.get_stats()['vector_dimension']}d vectors")
        else:
            print("⚠️  Transformer dependencies not available - TF-IDF only")
    
    except Exception as e:
        print(f"⚠️  Sentence-BERT failed to initialize: {e}")
        print("🔄 Continuing with TF-IDF comparison only...")
    
    # Test queries
    test_queries = [
        "cloud computing services",
        "software development consulting", 
        "office supplies and equipment",
        "business travel expenses"
    ]
    
    print(f"\n🎯 Comparing Search Methods")
    print("=" * 70)
    
    for i, query_text in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: \"{query_text}\"")
        print("-" * 50)
        
        # TF-IDF search
        start_time = asyncio.get_event_loop().time()
        tfidf_results = await tfidf_engine.search(query_text, top_k=5)
        tfidf_time = asyncio.get_event_loop().time() - start_time
        
        print(f"\n📊 TF-IDF Results:")
        print(f"   ⏱️  Search time: {tfidf_time:.3f}s")
        print(f"   📄 Results found: {len(tfidf_results)}")
        
        if tfidf_results:
            print(f"   🎯 Top score: {max(score for _, score in tfidf_results):.3f}")
            
            for j, (doc, score) in enumerate(tfidf_results[:3], 1):
                entity = doc.entity_id.replace('_', ' ').title()
                content_preview = doc.content[:80].replace('\n', ' ').strip()
                print(f"   {j}. [{doc.doc_type.value}] {entity} (score: {score:.3f})")
                print(f"      \"{content_preview}...\"")
        
        # Sentence-BERT search
        if transformer_engine:
            try:
                start_time = asyncio.get_event_loop().time()
                transformer_results = await transformer_engine.search(query_text, top_k=5)
                transformer_time = asyncio.get_event_loop().time() - start_time
                
                print(f"\n🤖 Sentence-BERT Results:")
                print(f"   ⏱️  Search time: {transformer_time:.3f}s")
                print(f"   📄 Results found: {len(transformer_results)}")
                
                if transformer_results:
                    print(f"   🎯 Top score: {max(score for _, score in transformer_results):.3f}")
                    
                    for j, (doc, score) in enumerate(transformer_results[:3], 1):
                        entity = doc.entity_id.replace('_', ' ').title()
                        content_preview = doc.content[:80].replace('\n', ' ').strip()
                        print(f"   {j}. [{doc.doc_type.value}] {entity} (score: {score:.3f})")
                        print(f"      \"{content_preview}...\"")
                
                # Comparison
                print(f"\n📈 Speed Comparison:")
                speedup = transformer_time / tfidf_time if tfidf_time > 0 else 0
                print(f"   TF-IDF is {speedup:.1f}x faster than Sentence-BERT")
                
            except Exception as e:
                print(f"   ❌ Sentence-BERT search failed: {e}")
        else:
            print(f"\n⚠️  Sentence-BERT not available for comparison")
    
    # Summary
    print(f"\n📊 Summary")
    print("=" * 70)
    print(f"✅ TF-IDF (Baseline):")
    print(f"   • Fast keyword-based search (1-10ms)")
    print(f"   • Good for exact term matching")
    print(f"   • Interpretable results")
    
    if transformer_engine:
        print(f"✅ Sentence-BERT (SOTA):")
        print(f"   • Semantic understanding (100-500ms)")
        print(f"   • Better for meaning-based queries")
        print(f"   • Handles synonyms and context")
        
        print(f"\n🎯 Key Insight:")
        print(f"   TF-IDF excels at exact keyword matching")
        print(f"   Sentence-BERT excels at semantic understanding")
        print(f"   Both have their place in production systems")
    else:
        print(f"⚠️  Sentence-BERT comparison unavailable")
        print(f"   Install with: pip install sentence-transformers torch faiss-cpu")
    
    print(f"\n✅ Comparison completed!")


async def main():
    """Main entry point."""
    try:
        await compare_search_methods()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
