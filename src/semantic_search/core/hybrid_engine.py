"""Hybrid search engine combining TF-IDF and Transformer approaches."""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from ..models.document import Document
from ..models.query import Query
from ..models.result import SearchResult
from ..utils.validators import validate_document, validate_query, validate_documents_batch
from ..utils.text_processing import TextProcessor
from .embeddings import TFIDFEmbedding
from .transformer_embeddings import TransformerEmbedding, TRANSFORMERS_AVAILABLE
from .exceptions import SearchError, DocumentProcessingError, ValidationError

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """
    Hybrid search engine combining TF-IDF and Transformer approaches.
    
    Provides both traditional keyword-based search and modern semantic search
    with intelligent result fusion and performance comparison capabilities.
    """
    
    def __init__(
        self,
        index_path: Optional[Path] = None,
        max_features: int = 10000,
        similarity_threshold: float = 0.1,
        max_workers: int = 4,
        transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_transformers: bool = True,
        fusion_method: str = "weighted_sum",
        tfidf_weight: float = 0.4,
        transformer_weight: float = 0.6
    ):
        """
        Initialize hybrid search engine.
        
        Args:
            index_path: Path to save/load index files
            max_features: Maximum TF-IDF features
            similarity_threshold: Default similarity threshold
            max_workers: Number of worker threads
            transformer_model: Transformer model name
            use_transformers: Whether to use transformer embeddings
            fusion_method: Method to combine scores ('weighted_sum', 'max', 'rank_fusion')
            tfidf_weight: Weight for TF-IDF scores in fusion
            transformer_weight: Weight for transformer scores in fusion
        """
        self.index_path = index_path or Path("./hybrid_index")
        self.similarity_threshold = similarity_threshold
        self.fusion_method = fusion_method
        self.tfidf_weight = tfidf_weight
        self.transformer_weight = transformer_weight
        
        # Initialize components
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.text_processor = TextProcessor()
        
        # Initialize TF-IDF engine (always available)
        self.tfidf_engine = TFIDFEmbedding(
            max_features=max_features,
            executor=self.executor
        )
        
        # Initialize transformer engine (if available and requested)
        self.transformer_engine = None
        self.use_transformers = use_transformers and TRANSFORMERS_AVAILABLE
        
        if self.use_transformers:
            try:
                self.transformer_engine = TransformerEmbedding(
                    model_name=transformer_model,
                    executor=self.executor
                )
                logger.info("Initialized hybrid engine with both TF-IDF and Transformers")
            except Exception as e:
                logger.warning(f"Failed to initialize transformer engine: {e}")
                self.use_transformers = False
                logger.info("Falling back to TF-IDF only")
        else:
            logger.info("Initialized hybrid engine with TF-IDF only")
        
        # State tracking
        self._stats = {
            'total_documents': 0,
            'total_searches': 0,
            'tfidf_searches': 0,
            'transformer_searches': 0,
            'hybrid_searches': 0,
            'avg_search_time': 0.0,
            'avg_tfidf_time': 0.0,
            'avg_transformer_time': 0.0
        }
        
        logger.info("Hybrid search engine initialized")
    
    async def add_document(self, document: Document) -> None:
        """Add a single document to both indices."""
        validate_document(document)
        await self.add_documents([document])
    
    async def add_documents(self, documents: List[Document]) -> None:
        """
        Add multiple documents to both search indices.
        
        Args:
            documents: List of documents to add
            
        Raises:
            ValidationError: If any document is invalid
            DocumentProcessingError: If processing fails
        """
        try:
            # Validate all documents first
            validate_documents_batch(documents)
            
            # Add to both engines in parallel
            tasks = []
            
            # Always add to TF-IDF
            tasks.append(self.tfidf_engine.add_documents(documents))
            
            # Add to transformer engine if available
            if self.use_transformers and self.transformer_engine:
                tasks.append(self.transformer_engine.add_documents(documents))
            
            # Execute in parallel
            await asyncio.gather(*tasks)
            
            # Update stats
            self._stats['total_documents'] += len(documents)
            
            logger.info(f"Successfully added {len(documents)} documents to hybrid index")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise DocumentProcessingError(f"Failed to add documents: {e}")
    
    async def search(
        self, 
        query: Query, 
        search_mode: str = "hybrid"
    ) -> List[SearchResult]:
        """
        Search using the specified mode.
        
        Args:
            query: Search query with filters
            search_mode: Search mode ('tfidf', 'transformer', 'hybrid')
            
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
            
            # Route to appropriate search method
            if search_mode == "tfidf":
                results = await self._search_tfidf_only(query)
                self._stats['tfidf_searches'] += 1
            elif search_mode == "transformer":
                if not self.use_transformers:
                    raise SearchError("Transformer search not available")
                results = await self._search_transformer_only(query)
                self._stats['transformer_searches'] += 1
            elif search_mode == "hybrid":
                results = await self._search_hybrid(query)
                self._stats['hybrid_searches'] += 1
            else:
                raise SearchError(f"Invalid search mode: {search_mode}")
            
            # Apply filters
            filtered_results = self._apply_filters(results, query)
            
            # Generate search results with context
            search_results = await self._generate_search_results(
                filtered_results, query, search_mode
            )
            
            # Limit to max results
            search_results = search_results[:query.max_results]
            
            # Update stats
            search_time = asyncio.get_event_loop().time() - start_time
            self._update_search_stats(search_time, search_mode)
            
            logger.info(
                f"{search_mode.upper()} search completed: "
                f"{len(search_results)} results in {search_time:.3f}s"
            )
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise SearchError(f"Search failed: {e}")
    
    async def _search_tfidf_only(self, query: Query) -> List[Tuple[Document, float]]:
        """Search using TF-IDF only."""
        start_time = asyncio.get_event_loop().time()
        
        results = await self.tfidf_engine.search(
            query.text,
            top_k=query.max_results * 2,
            threshold=query.similarity_threshold
        )
        
        search_time = asyncio.get_event_loop().time() - start_time
        self._stats['avg_tfidf_time'] = self._update_rolling_average(
            self._stats['avg_tfidf_time'], search_time, self._stats['tfidf_searches']
        )
        
        return results
    
    async def _search_transformer_only(self, query: Query) -> List[Tuple[Document, float]]:
        """Search using transformers only."""
        start_time = asyncio.get_event_loop().time()
        
        results = await self.transformer_engine.search(
            query.text,
            top_k=query.max_results * 2,
            threshold=query.similarity_threshold
        )
        
        search_time = asyncio.get_event_loop().time() - start_time
        self._stats['avg_transformer_time'] = self._update_rolling_average(
            self._stats['avg_transformer_time'], search_time, self._stats['transformer_searches']
        )
        
        return results
    
    async def _search_hybrid(self, query: Query) -> List[Tuple[Document, float]]:
        """Search using hybrid approach with result fusion."""
        # Get results from both engines in parallel
        tasks = [
            self.tfidf_engine.search(
                query.text,
                top_k=query.max_results * 3,
                threshold=query.similarity_threshold * 0.5  # Lower threshold for fusion
            )
        ]
        
        if self.use_transformers and self.transformer_engine:
            tasks.append(
                self.transformer_engine.search(
                    query.text,
                    top_k=query.max_results * 3,
                    threshold=query.similarity_threshold * 0.5
                )
            )
        
        results = await asyncio.gather(*tasks)
        tfidf_results = results[0]
        transformer_results = results[1] if len(results) > 1 else []
        
        # Fuse results
        fused_results = self._fuse_results(
            tfidf_results, 
            transformer_results, 
            query.max_results * 2
        )
        
        return fused_results
    
    def _fuse_results(
        self,
        tfidf_results: List[Tuple[Document, float]],
        transformer_results: List[Tuple[Document, float]],
        max_results: int
    ) -> List[Tuple[Document, float]]:
        """Fuse results from TF-IDF and transformer searches."""
        try:
            if self.fusion_method == "weighted_sum":
                return self._weighted_sum_fusion(tfidf_results, transformer_results, max_results)
            elif self.fusion_method == "max":
                return self._max_fusion(tfidf_results, transformer_results, max_results)
            elif self.fusion_method == "rank_fusion":
                return self._rank_fusion(tfidf_results, transformer_results, max_results)
            else:
                logger.warning(f"Unknown fusion method: {self.fusion_method}, using weighted_sum")
                return self._weighted_sum_fusion(tfidf_results, transformer_results, max_results)
        except Exception as e:
            logger.error(f"Result fusion failed: {e}")
            # Fallback to TF-IDF results
            return tfidf_results[:max_results]
    
    def _weighted_sum_fusion(
        self,
        tfidf_results: List[Tuple[Document, float]],
        transformer_results: List[Tuple[Document, float]],
        max_results: int
    ) -> List[Tuple[Document, float]]:
        """Fuse results using weighted sum of scores."""
        # Create document score maps
        tfidf_scores = {doc.id: score for doc, score in tfidf_results}
        transformer_scores = {doc.id: score for doc, score in transformer_results}
        
        # Get all unique documents
        all_doc_ids = set(tfidf_scores.keys()) | set(transformer_scores.keys())
        
        # Calculate fused scores
        fused_results = []
        for doc_id in all_doc_ids:
            tfidf_score = tfidf_scores.get(doc_id, 0.0)
            transformer_score = transformer_scores.get(doc_id, 0.0)
            
            # Weighted combination
            fused_score = (
                self.tfidf_weight * tfidf_score + 
                self.transformer_weight * transformer_score
            )
            
            # Find the document object
            doc = None
            for d, _ in tfidf_results:
                if d.id == doc_id:
                    doc = d
                    break
            if doc is None:
                for d, _ in transformer_results:
                    if d.id == doc_id:
                        doc = d
                        break
            
            if doc:
                fused_results.append((doc, fused_score))
        
        # Sort by fused score and return top results
        fused_results.sort(key=lambda x: x[1], reverse=True)
        return fused_results[:max_results]
    
    def _max_fusion(
        self,
        tfidf_results: List[Tuple[Document, float]],
        transformer_results: List[Tuple[Document, float]],
        max_results: int
    ) -> List[Tuple[Document, float]]:
        """Fuse results using maximum score."""
        # Create document score maps
        tfidf_scores = {doc.id: score for doc, score in tfidf_results}
        transformer_scores = {doc.id: score for doc, score in transformer_results}
        
        # Get all unique documents
        all_doc_ids = set(tfidf_scores.keys()) | set(transformer_scores.keys())
        
        # Calculate max scores
        fused_results = []
        for doc_id in all_doc_ids:
            tfidf_score = tfidf_scores.get(doc_id, 0.0)
            transformer_score = transformer_scores.get(doc_id, 0.0)
            
            # Maximum score
            max_score = max(tfidf_score, transformer_score)
            
            # Find the document object
            doc = None
            for d, _ in tfidf_results:
                if d.id == doc_id:
                    doc = d
                    break
            if doc is None:
                for d, _ in transformer_results:
                    if d.id == doc_id:
                        doc = d
                        break
            
            if doc:
                fused_results.append((doc, max_score))
        
        # Sort by max score and return top results
        fused_results.sort(key=lambda x: x[1], reverse=True)
        return fused_results[:max_results]
    
    def _rank_fusion(
        self,
        tfidf_results: List[Tuple[Document, float]],
        transformer_results: List[Tuple[Document, float]],
        max_results: int
    ) -> List[Tuple[Document, float]]:
        """Fuse results using reciprocal rank fusion."""
        # Create rank maps
        tfidf_ranks = {doc.id: i + 1 for i, (doc, _) in enumerate(tfidf_results)}
        transformer_ranks = {doc.id: i + 1 for i, (doc, _) in enumerate(transformer_results)}
        
        # Get all unique documents
        all_doc_ids = set(tfidf_ranks.keys()) | set(transformer_ranks.keys())
        
        # Calculate reciprocal rank fusion scores
        k = 60  # RRF parameter
        fused_results = []
        
        for doc_id in all_doc_ids:
            tfidf_rank = tfidf_ranks.get(doc_id, len(tfidf_results) + 1)
            transformer_rank = transformer_ranks.get(doc_id, len(transformer_results) + 1)
            
            # Reciprocal rank fusion score
            rrf_score = (1.0 / (k + tfidf_rank)) + (1.0 / (k + transformer_rank))
            
            # Find the document object
            doc = None
            for d, _ in tfidf_results:
                if d.id == doc_id:
                    doc = d
                    break
            if doc is None:
                for d, _ in transformer_results:
                    if d.id == doc_id:
                        doc = d
                        break
            
            if doc:
                fused_results.append((doc, rrf_score))
        
        # Sort by RRF score and return top results
        fused_results.sort(key=lambda x: x[1], reverse=True)
        return fused_results[:max_results]
    
    def _apply_filters(
        self, 
        results: List[Tuple[Document, float]], 
        query: Query
    ) -> List[Tuple[Document, float]]:
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
        results: List[Tuple[Document, float]],
        query: Query,
        search_mode: str
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
                
                # Create search result with additional metadata
                result = SearchResult(
                    document=document,
                    score=score,
                    context_snippet=context_snippet,
                    matched_terms=matched_terms,
                    rank=rank
                )
                
                # Add search mode to metadata
                result.document.metadata['search_mode'] = search_mode
                
                search_results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to generate result for document {document.id}: {e}")
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
    
    def _update_rolling_average(self, current_avg: float, new_value: float, count: int) -> float:
        """Update rolling average with new value."""
        if count <= 1:
            return new_value
        return (current_avg * (count - 1) + new_value) / count
    
    def _update_search_stats(self, search_time: float, search_mode: str) -> None:
        """Update search performance statistics."""
        self._stats['total_searches'] += 1
        
        # Update overall average
        total_searches = self._stats['total_searches']
        current_avg = self._stats['avg_search_time']
        self._stats['avg_search_time'] = (
            (current_avg * (total_searches - 1) + search_time) / total_searches
        )
    
    async def save_index(self, path: Optional[Path] = None) -> None:
        """Save all indices to disk."""
        save_path = path or self.index_path
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save TF-IDF index
        await self.tfidf_engine.save_index(save_path / "tfidf_index.pkl")
        
        # Save transformer index if available
        if self.use_transformers and self.transformer_engine:
            await self.transformer_engine.save_index(save_path / "transformer_index.pkl")
        
        logger.info(f"Hybrid index saved to {save_path}")
    
    async def load_index(self, path: Optional[Path] = None) -> None:
        """Load all indices from disk."""
        load_path = path or self.index_path
        
        # Load TF-IDF index
        tfidf_path = load_path / "tfidf_index.pkl"
        if tfidf_path.exists():
            await self.tfidf_engine.load_index(tfidf_path)
        
        # Load transformer index if available
        if self.use_transformers and self.transformer_engine:
            transformer_path = load_path / "transformer_index.pkl"
            if transformer_path.exists():
                await self.transformer_engine.load_index(transformer_path)
        
        # Update stats
        tfidf_stats = self.tfidf_engine.get_stats()
        self._stats['total_documents'] = tfidf_stats['total_documents']
        
        logger.info(f"Hybrid index loaded from {load_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics."""
        tfidf_stats = self.tfidf_engine.get_stats()
        
        stats = {
            **self._stats,
            'similarity_threshold': self.similarity_threshold,
            'fusion_method': self.fusion_method,
            'tfidf_weight': self.tfidf_weight,
            'transformer_weight': self.transformer_weight,
            'use_transformers': self.use_transformers,
            'index_path': str(self.index_path),
            'tfidf_stats': tfidf_stats
        }
        
        if self.use_transformers and self.transformer_engine:
            transformer_stats = self.transformer_engine.get_stats()
            stats['transformer_stats'] = transformer_stats
        
        return stats
    
    async def compare_search_methods(
        self, 
        query: Query
    ) -> Dict[str, Any]:
        """
        Compare all available search methods for a query.
        
        Args:
            query: Search query to compare
            
        Returns:
            Dictionary with results and performance metrics for each method
        """
        comparison = {
            'query': query.text,
            'methods': {}
        }
        
        # Test TF-IDF
        try:
            start_time = asyncio.get_event_loop().time()
            tfidf_results = await self.search(query, search_mode="tfidf")
            tfidf_time = asyncio.get_event_loop().time() - start_time
            
            comparison['methods']['tfidf'] = {
                'results_count': len(tfidf_results),
                'search_time': tfidf_time,
                'top_scores': [r.score for r in tfidf_results[:5]],
                'error': None
            }
        except Exception as e:
            comparison['methods']['tfidf'] = {'error': str(e)}
        
        # Test Transformer (if available)
        if self.use_transformers:
            try:
                start_time = asyncio.get_event_loop().time()
                transformer_results = await self.search(query, search_mode="transformer")
                transformer_time = asyncio.get_event_loop().time() - start_time
                
                comparison['methods']['transformer'] = {
                    'results_count': len(transformer_results),
                    'search_time': transformer_time,
                    'top_scores': [r.score for r in transformer_results[:5]],
                    'error': None
                }
            except Exception as e:
                comparison['methods']['transformer'] = {'error': str(e)}
        
        # Test Hybrid
        try:
            start_time = asyncio.get_event_loop().time()
            hybrid_results = await self.search(query, search_mode="hybrid")
            hybrid_time = asyncio.get_event_loop().time() - start_time
            
            comparison['methods']['hybrid'] = {
                'results_count': len(hybrid_results),
                'search_time': hybrid_time,
                'top_scores': [r.score for r in hybrid_results[:5]],
                'error': None
            }
        except Exception as e:
            comparison['methods']['hybrid'] = {'error': str(e)}
        
        return comparison
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            health = {
                'status': 'healthy',
                'engines': {},
                'stats': self.get_stats(),
                'timestamp': asyncio.get_event_loop().time()
            }
            
            # Check TF-IDF engine
            health['engines']['tfidf'] = {
                'available': True,
                'ready': self.tfidf_engine.is_fitted
            }
            
            # Check transformer engine
            health['engines']['transformer'] = {
                'available': self.use_transformers,
                'ready': (
                    self.transformer_engine.is_fitted 
                    if self.use_transformers else False
                )
            }
            
            # Overall readiness
            health['ready'] = self.tfidf_engine.is_fitted
            
            return health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': asyncio.get_event_loop().time()
            }
    
    async def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        logger.info("Hybrid search engine closed")
