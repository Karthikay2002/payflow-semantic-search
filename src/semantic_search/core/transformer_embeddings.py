"""Transformer-based embeddings for semantic search."""

import asyncio
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..models.document import Document
from ..utils.text_processing import TextProcessor
from .exceptions import IndexError, DocumentProcessingError

logger = logging.getLogger(__name__)


class TransformerEmbedding:
    """
    Transformer-based embedding system using Sentence-BERT and FinBERT.
    
    Provides state-of-the-art semantic embeddings with vector database
    support for high-performance similarity search.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_financial_model: bool = True,
        vector_dim: int = 384,
        index_type: str = "IVFFlat",
        nlist: int = 100,
        executor: Optional[ThreadPoolExecutor] = None,
        device: Optional[str] = None
    ):
        """
        Initialize transformer embedding system.
        
        Args:
            model_name: Sentence transformer model name
            use_financial_model: Whether to use financial domain model
            vector_dim: Dimension of embedding vectors
            index_type: FAISS index type (IVFFlat, HNSW, Flat)
            nlist: Number of clusters for IVF index
            executor: Thread pool executor for async operations
            device: Device to run models on ('cpu', 'cuda', 'mps')
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformer dependencies not available. Install with: "
                "pip install sentence-transformers torch faiss-cpu"
            )
        
        self.model_name = model_name
        self.use_financial_model = use_financial_model
        self.vector_dim = vector_dim
        self.index_type = index_type
        self.nlist = nlist
        self.device = device or self._get_best_device()
        
        # Initialize models
        self.primary_model = None
        self.financial_model = None
        self._load_models()
        
        # Vector database
        self.faiss_index = None
        self.document_index: Dict[str, int] = {}
        self.documents: List[Document] = []
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self.is_fitted = False
        
        # Text processor and executor
        self.text_processor = TextProcessor()
        self._executor = executor or ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Initialized transformer embedding with {self.model_name} on {self.device}")
    
    def _get_best_device(self) -> str:
        """Determine the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_models(self) -> None:
        """Load transformer models."""
        try:
            # Load primary model
            self.primary_model = SentenceTransformer(self.model_name, device=self.device)
            
            # Load financial domain model if requested
            if self.use_financial_model:
                try:
                    # Use FinBERT or finance-specific sentence transformer
                    financial_models = [
                        "sentence-transformers/all-mpnet-base-v2",  # Good general model
                        "microsoft/DialoGPT-medium",  # Alternative
                        "sentence-transformers/paraphrase-MiniLM-L6-v2"  # Fallback
                    ]
                    
                    for model_name in financial_models:
                        try:
                            self.financial_model = SentenceTransformer(model_name, device=self.device)
                            logger.info(f"Loaded financial model: {model_name}")
                            break
                        except Exception as e:
                            logger.warning(f"Failed to load {model_name}: {e}")
                            continue
                    
                    if self.financial_model is None:
                        logger.warning("No financial model loaded, using primary model only")
                        self.financial_model = self.primary_model
                        
                except Exception as e:
                    logger.warning(f"Failed to load financial model: {e}")
                    self.financial_model = self.primary_model
            else:
                self.financial_model = self.primary_model
                
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise DocumentProcessingError(f"Failed to load transformer models: {e}")
    
    async def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector index asynchronously.
        
        Args:
            documents: List of documents to add
            
        Raises:
            DocumentProcessingError: If document processing fails
            IndexError: If index update fails
        """
        try:
            logger.info(f"Adding {len(documents)} documents to transformer index")
            
            # Process documents and generate embeddings
            embeddings = await self._generate_embeddings(documents)
            
            # Update index
            await self._update_vector_index(documents, embeddings)
            
            logger.info(f"Successfully added {len(documents)} documents to transformer index")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise DocumentProcessingError(f"Failed to add documents: {e}")
    
    async def _generate_embeddings(self, documents: List[Document]) -> np.ndarray:
        """Generate embeddings for documents."""
        try:
            # Prepare texts for embedding
            texts = []
            for doc in documents:
                # Enhanced text preprocessing for financial documents
                processed_text = self._prepare_financial_text(doc)
                texts.append(processed_text)
            
            # Generate embeddings in thread pool
            embeddings = await asyncio.get_event_loop().run_in_executor(
                self._executor, self._encode_texts, texts, documents
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise DocumentProcessingError(f"Failed to generate embeddings: {e}")
    
    def _prepare_financial_text(self, document: Document) -> str:
        """Prepare financial document text for embedding."""
        try:
            # Clean the text
            text = self.text_processor.clean_text(document.content)
            
            # Add document type context for better embeddings
            type_context = f"[{document.doc_type.value.upper()}]"
            
            # Add entity context
            entity_context = f"[ENTITY: {document.entity_id}]"
            
            # Truncate to reasonable length (transformers have token limits)
            max_length = 512  # tokens, roughly 2048 characters
            if len(text) > max_length * 4:  # rough character estimate
                # Extract key sections for financial documents
                text = self._extract_key_sections(text, document.doc_type)
                text = text[:max_length * 4]
            
            # Combine context and content
            prepared_text = f"{type_context} {entity_context} {text}"
            
            return prepared_text
            
        except Exception as e:
            logger.warning(f"Failed to prepare text for document {document.id}: {e}")
            return self.text_processor.clean_text(document.content)[:2000]
    
    def _extract_key_sections(self, text: str, doc_type) -> str:
        """Extract key sections from financial documents."""
        try:
            # Financial keywords to prioritize
            financial_keywords = [
                'revenue', 'earnings', 'profit', 'loss', 'cash flow', 'assets', 'liabilities',
                'equity', 'debt', 'margin', 'growth', 'dividend', 'eps', 'ebitda',
                'operating income', 'net income', 'gross profit', 'balance sheet',
                'income statement', 'financial position', 'liquidity', 'capital',
                'investment', 'acquisition', 'merger', 'outlook', 'guidance', 'forecast'
            ]
            
            # Find sentences containing financial keywords
            sentences = text.split('.')
            key_sentences = []
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in financial_keywords):
                    key_sentences.append(sentence.strip())
            
            if key_sentences:
                # Return top sentences up to length limit
                result = '. '.join(key_sentences)
                if len(result) > 2000:
                    result = result[:2000] + '...'
                return result
            else:
                # Fallback to first portion
                return text[:2000]
                
        except Exception as e:
            logger.warning(f"Failed to extract key sections: {e}")
            return text[:2000]
    
    def _encode_texts(self, texts: List[str], documents: List[Document]) -> np.ndarray:
        """Encode texts using transformer models."""
        try:
            # Use primary model for all documents to ensure consistent dimensions
            embeddings = self.primary_model.encode(texts, convert_to_numpy=True)
            
            # Cache embeddings
            for i, doc in enumerate(documents):
                self.embeddings_cache[doc.id] = embeddings[i]
            
            return embeddings.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise
    
    def _is_financial_document(self, document: Document) -> bool:
        """Check if document should use financial model."""
        financial_types = {
            'annual_report', 'quarterly_report', 'current_report',
            'earnings_transcript', 'financial_statement', 'invoice',
            'purchase_order', 'contract'
        }
        return document.doc_type.value in financial_types
    
    async def _update_vector_index(self, documents: List[Document], embeddings: np.ndarray) -> None:
        """Update FAISS vector index with new documents and embeddings."""
        try:
            # Add documents to collection
            start_idx = len(self.documents)
            self.documents.extend(documents)
            
            # Update document index mapping
            for i, doc in enumerate(documents):
                self.document_index[doc.id] = start_idx + i
            
            # Initialize or update FAISS index
            if self.faiss_index is None:
                self._initialize_faiss_index(embeddings.shape[1])
            
            # Add embeddings to index
            await asyncio.get_event_loop().run_in_executor(
                self._executor, self._add_to_faiss_index, embeddings
            )
            
            self.is_fitted = True
            
        except Exception as e:
            logger.error(f"Failed to update vector index: {e}")
            raise IndexError(f"Failed to update vector index: {e}")
    
    def _initialize_faiss_index(self, dimension: int) -> None:
        """Initialize FAISS index."""
        try:
            if self.index_type == "Flat":
                self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            elif self.index_type == "IVFFlat":
                quantizer = faiss.IndexFlatIP(dimension)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist)
            elif self.index_type == "HNSW":
                self.faiss_index = faiss.IndexHNSWFlat(dimension, 32)
                self.faiss_index.hnsw.efConstruction = 200
                self.faiss_index.hnsw.efSearch = 100
            else:
                logger.warning(f"Unknown index type {self.index_type}, using Flat")
                self.faiss_index = faiss.IndexFlatIP(dimension)
            
            logger.info(f"Initialized FAISS index: {self.index_type} with dimension {dimension}")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            raise
    
    def _add_to_faiss_index(self, embeddings: np.ndarray) -> None:
        """Add embeddings to FAISS index (synchronous)."""
        try:
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Train index if needed (for IVF)
            if hasattr(self.faiss_index, 'is_trained') and not self.faiss_index.is_trained:
                # Need sufficient data to train IVF
                if len(self.documents) >= self.nlist * 2:
                    all_embeddings = np.array([
                        self.embeddings_cache[doc.id] for doc in self.documents
                        if doc.id in self.embeddings_cache
                    ], dtype=np.float32)
                    faiss.normalize_L2(all_embeddings)
                    self.faiss_index.train(all_embeddings)
                    logger.info("Trained FAISS IVF index")
                else:
                    # Switch to flat index for small datasets
                    logger.info("Switching to Flat index for small dataset")
                    self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
            
            # Add to index
            self.faiss_index.add(embeddings)
            
        except Exception as e:
            logger.error(f"Failed to add to FAISS index: {e}")
            raise
    
    async def search(
        self, 
        query: str, 
        top_k: int = 50,
        threshold: float = 0.1
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents using transformer embeddings.
        
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
            # Generate query embedding
            query_embedding = await self._generate_query_embedding(query)
            
            # Search in FAISS index
            results = await asyncio.get_event_loop().run_in_executor(
                self._executor, self._search_faiss, query_embedding, top_k, threshold
            )
            
            logger.debug(f"Transformer search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Transformer search failed: {e}")
            raise DocumentProcessingError(f"Transformer search failed: {e}")
    
    async def _generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for search query."""
        try:
            # Clean query
            processed_query = self.text_processor.clean_text(query)
            
            # Add query context
            query_context = "[SEARCH QUERY]"
            prepared_query = f"{query_context} {processed_query}"
            
            # Generate embedding
            embedding = await asyncio.get_event_loop().run_in_executor(
                self._executor, self._encode_query, prepared_query
            )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise
    
    def _encode_query(self, query: str) -> np.ndarray:
        """Encode query using primary model."""
        embedding = self.primary_model.encode([query], convert_to_numpy=True)[0]
        return embedding.astype(np.float32)
    
    def _search_faiss(
        self, 
        query_embedding: np.ndarray, 
        top_k: int, 
        threshold: float
    ) -> List[Tuple[Document, float]]:
        """Search FAISS index (synchronous)."""
        try:
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.faiss_index.search(query_embedding, top_k)
            
            # Process results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and score >= threshold:  # Valid index and above threshold
                    document = self.documents[idx]
                    results.append((document, float(score)))
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
    
    async def save_index(self, path: Path) -> None:
        """Save the transformer index to disk."""
        try:
            index_data = {
                'faiss_index': self.faiss_index,
                'document_index': self.document_index,
                'documents': self.documents,
                'embeddings_cache': self.embeddings_cache,
                'is_fitted': self.is_fitted,
                'config': {
                    'model_name': self.model_name,
                    'use_financial_model': self.use_financial_model,
                    'vector_dim': self.vector_dim,
                    'index_type': self.index_type,
                    'nlist': self.nlist
                }
            }
            
            # Save in thread pool
            await asyncio.get_event_loop().run_in_executor(
                self._executor, self._save_index_sync, path, index_data
            )
            
            logger.info(f"Transformer index saved to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save transformer index: {e}")
            raise IndexError(f"Failed to save transformer index: {e}")
    
    def _save_index_sync(self, path: Path, index_data: Dict[str, Any]) -> None:
        """Save index synchronously."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index separately
        if index_data['faiss_index'] is not None:
            faiss_path = path.parent / f"{path.stem}_faiss.index"
            faiss.write_index(index_data['faiss_index'], str(faiss_path))
            index_data['faiss_index_path'] = str(faiss_path)
            index_data['faiss_index'] = None  # Don't pickle FAISS index
        
        # Save the rest
        with open(path, 'wb') as f:
            pickle.dump(index_data, f)
    
    async def load_index(self, path: Path) -> None:
        """Load the transformer index from disk."""
        try:
            # Load in thread pool
            index_data = await asyncio.get_event_loop().run_in_executor(
                self._executor, self._load_index_sync, path
            )
            
            # Restore state
            self.document_index = index_data['document_index']
            self.documents = index_data['documents']
            self.embeddings_cache = index_data['embeddings_cache']
            self.is_fitted = index_data['is_fitted']
            
            # Load FAISS index if it exists
            if 'faiss_index_path' in index_data:
                faiss_path = Path(index_data['faiss_index_path'])
                if faiss_path.exists():
                    self.faiss_index = faiss.read_index(str(faiss_path))
                else:
                    logger.warning("FAISS index file not found, will rebuild on next add")
                    self.faiss_index = None
                    self.is_fitted = False
            
            logger.info(f"Transformer index loaded from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load transformer index: {e}")
            raise IndexError(f"Failed to load transformer index: {e}")
    
    def _load_index_sync(self, path: Path) -> Dict[str, Any]:
        """Load index synchronously."""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transformer index statistics."""
        return {
            'total_documents': len(self.documents),
            'vector_dimension': self.vector_dim,
            'index_type': self.index_type,
            'is_fitted': self.is_fitted,
            'model_name': self.model_name,
            'use_financial_model': self.use_financial_model,
            'device': self.device,
            'cached_embeddings': len(self.embeddings_cache),
            'faiss_index_size': self.faiss_index.ntotal if self.faiss_index else 0
        }
