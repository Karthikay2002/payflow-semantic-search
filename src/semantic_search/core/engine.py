"""Main semantic search engine implementation."""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from ..models.document import Document, DocumentType
from ..models.query import Query, DateRange
from ..models.result import SearchResult
from ..utils.validators import validate_document, validate_query, validate_documents_batch
from ..utils.text_processing import TextProcessor
from .embeddings import TFIDFEmbedding
from .exceptions import SearchError, DocumentProcessingError, ValidationError

logger = logging.getLogger(__name__)


class SemanticSearchEngine:
    """
    Production-ready semantic search engine for financial documents.
    
    Provides async document indexing, filtering, and similarity search
    with comprehensive error handling and performance monitoring.
    """
    
    def __init__(
        self,
        index_path: Optional[Path] = None,
        max_features: int = 10000,
        similarity_threshold: float = 0.1,
        max_workers: int = 4
    ):
        """
        Initialize semantic search engine.
        
        Args:
            index_path: Path to save/load index files
            max_features: Maximum TF-IDF features
            similarity_threshold: Default similarity threshold
            max_workers: Number of worker threads
        """
        self.index_path = index_path or Path("./index")
        self.similarity_threshold = similarity_threshold
        
        # Initialize components
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.embedding_engine = TFIDFEmbedding(
            max_features=max_features,
            executor=self.executor
        )
        self.text_processor = TextProcessor()
        
        # State tracking
        self._stats = {
            'total_documents': 0,
            'total_searches': 0,
            'avg_search_time': 0.0
        }
        
        logger.info("Semantic search engine initialized")
    
    async def add_document(self, document: Document) -> None:
        """
        Add a single document to the index.
        
        Args:
            document: Document to add
            
        Raises:
            ValidationError: If document is invalid
            DocumentProcessingError: If processing fails
        """
        validate_document(document)
        await self.add_documents([document])
    
    async def add_documents(self, documents: List[Document]) -> None:
        """
        Add multiple documents to the index asynchronously.
        
        Args:
            documents: List of documents to add
            
        Raises:
            ValidationError: If any document is invalid
            DocumentProcessingError: If processing fails
        """
        try:
            # Validate all documents first
            validate_documents_batch(documents)
            
            # Add to embedding engine
            await self.embedding_engine.add_documents(documents)
            
            # Update stats
            self._stats['total_documents'] += len(documents)
            
            logger.info(f"Successfully added {len(documents)} documents to index")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise DocumentProcessingError(f"Failed to add documents: {str(e)}")
    
    async def search(self, query: Query) -> List[SearchResult]:
        """
        Search for documents matching the query.
        
        Args:
            query: Search query with filters
            
        Returns:
            List of search results ranked by relevance
            
        Raises:
            ValidationError: If query is invalid
            SearchError: If search fails
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Validate query
            validate_query(query)
            
            # Get initial results from embedding engine
            embedding_results = await self.embedding_engine.search(
                query.text,
                top_k=query.max_results * 2,  # Get more to allow for filtering
                threshold=query.similarity_threshold
            )
            
            # Apply filters
            filtered_results = self._apply_filters(embedding_results, query)
            
            # Generate search results with context
            search_results = await self._generate_search_results(
                filtered_results, query
            )
            
            # Limit to max results
            search_results = search_results[:query.max_results]
            
            # Update stats
            search_time = asyncio.get_event_loop().time() - start_time
            self._update_search_stats(search_time)
            
            logger.info(f"Search completed: {len(search_results)} results in {search_time:.3f}s")
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise SearchError(f"Search failed: {str(e)}")
    
    def _apply_filters(
        self, 
        results: List[tuple[Document, float]], 
        query: Query
    ) -> List[tuple[Document, float]]:
        """Apply query filters to search results."""
        if not results:
            return results
        
        filtered = []
        
        for document, score in results:
            # Entity ID filter
            if query.entity_ids and document.entity_id not in query.entity_ids:
                continue
            
            # Document type filter
            if query.doc_types and document.doc_type not in query.doc_types:
                continue
            
            # Date range filter
            if query.date_range and not query.date_range.contains(document.date):
                continue
            
            filtered.append((document, score))
        
        return filtered
    
    async def _generate_search_results(
        self,
        results: List[tuple[Document, float]],
        query: Query
    ) -> List[SearchResult]:
        """Generate SearchResult objects with context snippets."""
        if not results:
            return []
        
        # Extract query terms for context generation
        query_terms = query.text.lower().split()
        
        search_results = []
        for rank, (document, score) in enumerate(results, 1):
            try:
                # Generate context snippet
                context_snippet = self.text_processor.generate_context_snippet(
                    document.content, query_terms, max_length=200
                )
                
                # Find matched terms
                matched_terms = self._find_matched_terms(document.content, query_terms)
                
                # Create search result
                result = SearchResult(
                    document=document,
                    score=score,
                    context_snippet=context_snippet,
                    matched_terms=matched_terms,
                    rank=rank
                )
                
                search_results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to generate result for document {document.id}: {str(e)}")
                continue
        
        return search_results
    
    def _find_matched_terms(self, content: str, query_terms: List[str]) -> List[str]:
        """Find query terms that appear in document content."""
        content_lower = content.lower()
        matched = []
        
        for term in query_terms:
            if term in content_lower:
                matched.append(term)
        
        # Also check for financial terms
        financial_terms = self.text_processor.extract_key_terms(content)
        for term in financial_terms:
            if any(query_term in term for query_term in query_terms):
                matched.append(term)
        
        return list(set(matched))  # Remove duplicates
    
    def _update_search_stats(self, search_time: float) -> None:
        """Update search performance statistics."""
        self._stats['total_searches'] += 1
        
        # Update rolling average
        total_searches = self._stats['total_searches']
        current_avg = self._stats['avg_search_time']
        self._stats['avg_search_time'] = (
            (current_avg * (total_searches - 1) + search_time) / total_searches
        )
    
    async def save_index(self, path: Optional[Path] = None) -> None:
        """Save the search index to disk."""
        save_path = path or self.index_path / "index.pkl"
        await self.embedding_engine.save_index(save_path)
        logger.info(f"Index saved to {save_path}")
    
    async def load_index(self, path: Optional[Path] = None) -> None:
        """Load the search index from disk."""
        load_path = path or self.index_path / "index.pkl"
        await self.embedding_engine.load_index(load_path)
        
        # Update stats
        stats = self.embedding_engine.get_stats()
        self._stats['total_documents'] = stats['total_documents']
        
        logger.info(f"Index loaded from {load_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        embedding_stats = self.embedding_engine.get_stats()
        
        return {
            **self._stats,
            **embedding_stats,
            'similarity_threshold': self.similarity_threshold,
            'index_path': str(self.index_path)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of the search engine."""
        try:
            # Check if index is ready
            is_ready = self.embedding_engine.is_fitted
            
            # Get stats
            stats = self.get_stats()
            
            return {
                'status': 'healthy' if is_ready else 'not_ready',
                'is_ready': is_ready,
                'stats': stats,
                'timestamp': asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': asyncio.get_event_loop().time()
            }
    
    async def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        logger.info("Semantic search engine closed")
