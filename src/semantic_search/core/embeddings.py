"""TF-IDF embedding implementation for semantic search."""

import asyncio
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..models.document import Document
from ..utils.text_processing import TextProcessor
from .exceptions import IndexError, DocumentProcessingError

logger = logging.getLogger(__name__)


class TFIDFEmbedding:
    """
    TF-IDF based embedding system for semantic search.
    
    Provides efficient document vectorization and similarity search
    with persistence and incremental updates.
    """
    
    def __init__(
        self,
        max_features: int = 10000,
        min_df: int = 1,  # Changed from 2 to 1 for small document sets
        max_df: float = 0.95,  # Changed from 0.8 to 0.95 for small document sets
        ngram_range: Tuple[int, int] = (1, 2),
        executor: Optional[ThreadPoolExecutor] = None
    ):
        """
        Initialize TF-IDF embedding system.
        
        Args:
            max_features: Maximum number of features to extract
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms
            ngram_range: N-gram range for feature extraction
            executor: Thread pool executor for async operations
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        
        # Adjust parameters based on expected document count
        # For small datasets, use more permissive settings
        adjusted_min_df = min_df
        adjusted_max_df = max_df
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=adjusted_min_df,
            max_df=adjusted_max_df,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            strip_accents='ascii',
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
        )
        
        self.text_processor = TextProcessor()
        self.document_vectors = None
        self.document_index: Dict[str, int] = {}
        self.documents: List[Document] = []
        self.is_fitted = False
        
        # Thread pool for async operations
        self._executor = executor or ThreadPoolExecutor(max_workers=4)
        
    async def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the index asynchronously.
        
        Args:
            documents: List of documents to add
            
        Raises:
            DocumentProcessingError: If document processing fails
            IndexError: If index update fails
        """
        try:
            # Process documents in thread pool
            processed_docs = await asyncio.get_event_loop().run_in_executor(
                self._executor, self._process_documents, documents
            )
            
            # Update index
            await self._update_index(processed_docs)
            
            logger.info(f"Added {len(documents)} documents to index")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise DocumentProcessingError(f"Failed to add documents: {str(e)}")
    
    async def search(
        self, 
        query: str, 
        top_k: int = 50,
        threshold: float = 0.1
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents asynchronously.
        
        Args:
            query: Search query text
            top_k: Maximum number of results
            threshold: Minimum similarity threshold
            
        Returns:
            List of (document, similarity_score) tuples
            
        Raises:
            IndexError: If index is not ready
            DocumentProcessingError: If query processing fails
        """
        if not self.is_fitted:
            raise IndexError("Index is not fitted. Add documents first.")
        
        try:
            # Process query and search in thread pool
            results = await asyncio.get_event_loop().run_in_executor(
                self._executor, self._search_sync, query, top_k, threshold
            )
            
            logger.debug(f"Search returned {len(results)} results for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise DocumentProcessingError(f"Search failed: {str(e)}")
    
    def _process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents synchronously in thread pool."""
        processed = []
        for doc in documents:
            try:
                # Clean and preprocess document content
                processed_content = self.text_processor.clean_text(doc.content)
                
                # Create processed document
                processed_doc = Document(
                    id=doc.id,
                    content=processed_content,
                    entity_id=doc.entity_id,
                    doc_type=doc.doc_type,
                    date=doc.date,
                    metadata=doc.metadata
                )
                processed.append(processed_doc)
                
            except Exception as e:
                logger.warning(f"Failed to process document {doc.id}: {str(e)}")
                continue
                
        return processed
    
    async def _update_index(self, documents: List[Document]) -> None:
        """Update the search index with new documents."""
        if not documents:
            return
            
        # Add documents to collection
        start_idx = len(self.documents)
        self.documents.extend(documents)
        
        # Update document index mapping
        for i, doc in enumerate(documents):
            self.document_index[doc.id] = start_idx + i
        
        # Refit vectorizer with all documents
        all_content = [doc.content for doc in self.documents]
        
        # Adjust parameters for small document sets
        if len(all_content) < 5:
            # For very small document sets, use minimal constraints
            self.vectorizer.set_params(min_df=1, max_df=1.0)
        elif len(all_content) < 20:
            # For small document sets, use relaxed constraints
            self.vectorizer.set_params(min_df=1, max_df=0.95)
        
        # Run vectorization in thread pool
        try:
            self.document_vectors = await asyncio.get_event_loop().run_in_executor(
                self._executor, self.vectorizer.fit_transform, all_content
            )
        except ValueError as e:
            if "max_df corresponds to < documents than min_df" in str(e) or "no terms remain" in str(e):
                # Fallback: use most permissive settings
                self.vectorizer.set_params(min_df=1, max_df=1.0)
                try:
                    self.document_vectors = await asyncio.get_event_loop().run_in_executor(
                        self._executor, self.vectorizer.fit_transform, all_content
                    )
                except ValueError as e2:
                    # Final fallback: even more permissive
                    self.vectorizer = TfidfVectorizer(
                        max_features=min(1000, len(all_content) * 100),
                        min_df=1,
                        max_df=1.0,
                        ngram_range=(1, 1),  # Unigrams only for small datasets
                        stop_words=None,  # Keep all words for small datasets
                        lowercase=True
                    )
                    self.document_vectors = await asyncio.get_event_loop().run_in_executor(
                        self._executor, self.vectorizer.fit_transform, all_content
                    )
            else:
                raise
        
        self.is_fitted = True
    
    def _search_sync(
        self, 
        query: str, 
        top_k: int, 
        threshold: float
    ) -> List[Tuple[Document, float]]:
        """Perform synchronous search in thread pool."""
        # Process and vectorize query
        processed_query = self.text_processor.clean_text(query)
        query_vector = self.vectorizer.transform([processed_query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        
        # Filter by threshold and get top results
        valid_indices = np.where(similarities >= threshold)[0]
        
        if len(valid_indices) == 0:
            return []
        
        # Sort by similarity score (descending)
        sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
        
        # Return top_k results
        results = []
        for idx in sorted_indices[:top_k]:
            document = self.documents[idx]
            score = float(similarities[idx])
            results.append((document, score))
        
        return results
    
    def get_feature_names(self) -> List[str]:
        """Get feature names from the vectorizer."""
        if not self.is_fitted:
            return []
        return self.vectorizer.get_feature_names_out().tolist()
    
    async def save_index(self, path: Path) -> None:
        """Save the index to disk asynchronously."""
        try:
            index_data = {
                'vectorizer': self.vectorizer,
                'document_vectors': self.document_vectors,
                'document_index': self.document_index,
                'documents': self.documents,
                'is_fitted': self.is_fitted,
                'config': {
                    'max_features': self.max_features,
                    'min_df': self.min_df,
                    'max_df': self.max_df,
                    'ngram_range': self.ngram_range
                }
            }
            
            # Save in thread pool
            await asyncio.get_event_loop().run_in_executor(
                self._executor, self._save_index_sync, path, index_data
            )
            
            logger.info(f"Index saved to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
            raise IndexError(f"Failed to save index: {str(e)}")
    
    def _save_index_sync(self, path: Path, index_data: Dict[str, Any]) -> None:
        """Save index synchronously in thread pool."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(index_data, f)
    
    async def load_index(self, path: Path) -> None:
        """Load the index from disk asynchronously."""
        try:
            # Load in thread pool
            index_data = await asyncio.get_event_loop().run_in_executor(
                self._executor, self._load_index_sync, path
            )
            
            # Restore state
            self.vectorizer = index_data['vectorizer']
            self.document_vectors = index_data['document_vectors']
            self.document_index = index_data['document_index']
            self.documents = index_data['documents']
            self.is_fitted = index_data['is_fitted']
            
            logger.info(f"Index loaded from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            raise IndexError(f"Failed to load index: {str(e)}")
    
    def _load_index_sync(self, path: Path) -> Dict[str, Any]:
        """Load index synchronously in thread pool."""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'total_documents': len(self.documents),
            'vocabulary_size': len(self.vectorizer.vocabulary_) if self.is_fitted else 0,
            'is_fitted': self.is_fitted,
            'config': {
                'max_features': self.max_features,
                'min_df': self.min_df,
                'max_df': self.max_df,
                'ngram_range': self.ngram_range
            }
        }
