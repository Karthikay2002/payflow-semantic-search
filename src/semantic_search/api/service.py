"""High-level API service for semantic search."""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncContextManager
from contextlib import asynccontextmanager

from ..core.engine import SemanticSearchEngine
from ..models.document import Document
from ..models.query import Query
from ..models.result import SearchResult
from ..utils.logging_config import setup_logging
from ..core.exceptions import SemanticSearchError

logger = logging.getLogger(__name__)


class SemanticSearchService:
    """
    High-level service interface for semantic search operations.
    
    Provides a clean, production-ready API for document indexing and search
    with proper resource management and error handling.
    """
    
    def __init__(
        self,
        index_path: Optional[Path] = None,
        max_features: int = 10000,
        similarity_threshold: float = 0.1,
        max_workers: int = 4,
        auto_save: bool = True,
        log_level: str = "INFO"
    ):
        """
        Initialize semantic search service.
        
        Args:
            index_path: Path for index persistence
            max_features: Maximum TF-IDF features
            similarity_threshold: Default similarity threshold
            max_workers: Number of worker threads
            auto_save: Whether to auto-save index after updates
            log_level: Logging level
        """
        # Setup logging
        setup_logging(level=log_level)
        
        self.index_path = index_path or Path("./semantic_search_index")
        self.auto_save = auto_save
        
        # Initialize engine
        self.engine = SemanticSearchEngine(
            index_path=self.index_path,
            max_features=max_features,
            similarity_threshold=similarity_threshold,
            max_workers=max_workers
        )
        
        self._initialized = False
        logger.info("Semantic search service initialized")
    
    async def initialize(self, load_existing: bool = True) -> None:
        """
        Initialize the service and optionally load existing index.
        
        Args:
            load_existing: Whether to load existing index from disk
        """
        try:
            if load_existing and (self.index_path / "index.pkl").exists():
                await self.engine.load_index()
                logger.info("Loaded existing index")
            
            self._initialized = True
            logger.info("Service initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize service: {str(e)}")
            raise SemanticSearchError(f"Service initialization failed: {str(e)}")
    
    async def add_document(self, document: Document) -> None:
        """
        Add a single document to the search index.
        
        Args:
            document: Document to add
            
        Raises:
            SemanticSearchError: If operation fails
        """
        self._check_initialized()
        
        try:
            await self.engine.add_document(document)
            
            if self.auto_save:
                await self.engine.save_index()
            
            logger.debug(f"Added document: {document.id}")
            
        except Exception as e:
            logger.error(f"Failed to add document {document.id}: {str(e)}")
            raise SemanticSearchError(f"Failed to add document: {str(e)}")
    
    async def add_documents(self, documents: List[Document]) -> None:
        """
        Add multiple documents to the search index.
        
        Args:
            documents: List of documents to add
            
        Raises:
            SemanticSearchError: If operation fails
        """
        self._check_initialized()
        
        try:
            await self.engine.add_documents(documents)
            
            if self.auto_save:
                await self.engine.save_index()
            
            logger.info(f"Added {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to add {len(documents)} documents: {str(e)}")
            raise SemanticSearchError(f"Failed to add documents: {str(e)}")
    
    async def search(self, query: Query) -> List[SearchResult]:
        """
        Search for documents matching the query.
        
        Args:
            query: Search query with filters
            
        Returns:
            List of ranked search results
            
        Raises:
            SemanticSearchError: If search fails
        """
        self._check_initialized()
        
        try:
            results = await self.engine.search(query)
            logger.debug(f"Search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise SemanticSearchError(f"Search failed: {str(e)}")
    
    async def search_text(
        self,
        text: str,
        entity_ids: Optional[List[str]] = None,
        doc_types: Optional[List[str]] = None,
        similarity_threshold: Optional[float] = None,
        max_results: int = 50
    ) -> List[SearchResult]:
        """
        Convenience method for simple text search.
        
        Args:
            text: Search text
            entity_ids: Optional entity ID filters
            doc_types: Optional document type filters
            similarity_threshold: Optional similarity threshold
            max_results: Maximum results to return
            
        Returns:
            List of search results
        """
        from ..models.document import DocumentType
        
        # Convert string doc_types to enum
        converted_doc_types = None
        if doc_types:
            converted_doc_types = {DocumentType(dt) for dt in doc_types}
        
        # Create query
        query = Query(
            text=text,
            entity_ids=set(entity_ids) if entity_ids else None,
            doc_types=converted_doc_types,
            similarity_threshold=similarity_threshold or self.engine.similarity_threshold,
            max_results=max_results
        )
        
        return await self.search(query)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get service and engine statistics."""
        self._check_initialized()
        
        engine_stats = self.engine.get_stats()
        
        return {
            'service': {
                'initialized': self._initialized,
                'auto_save': self.auto_save,
                'index_path': str(self.index_path)
            },
            'engine': engine_stats
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            if not self._initialized:
                return {
                    'status': 'not_initialized',
                    'message': 'Service not initialized'
                }
            
            return await self.engine.health_check()
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    async def save_index(self, path: Optional[Path] = None) -> None:
        """Save the search index to disk."""
        self._check_initialized()
        await self.engine.save_index(path)
    
    async def load_index(self, path: Optional[Path] = None) -> None:
        """Load the search index from disk."""
        self._check_initialized()
        await self.engine.load_index(path)
    
    def _check_initialized(self) -> None:
        """Check if service is properly initialized."""
        if not self._initialized:
            raise SemanticSearchError("Service not initialized. Call initialize() first.")
    
    async def close(self) -> None:
        """Clean up resources and close the service."""
        try:
            if self.auto_save and self._initialized:
                await self.engine.save_index()
            
            await self.engine.close()
            self._initialized = False
            logger.info("Service closed successfully")
            
        except Exception as e:
            logger.error(f"Error during service shutdown: {str(e)}")
    
    @classmethod
    @asynccontextmanager
    async def create(
        cls,
        index_path: Optional[Path] = None,
        **kwargs
    ) -> AsyncContextManager['SemanticSearchService']:
        """
        Create and manage service lifecycle with context manager.
        
        Args:
            index_path: Path for index persistence
            **kwargs: Additional service configuration
            
        Yields:
            Initialized semantic search service
        """
        service = cls(index_path=index_path, **kwargs)
        
        try:
            await service.initialize()
            yield service
        finally:
            await service.close()
