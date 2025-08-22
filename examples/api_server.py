"""
FastAPI server example for production deployment.

Provides REST API endpoints for the semantic search system
with proper error handling and OpenAPI documentation.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from semantic_search.api.service import SemanticSearchService
from semantic_search.models.document import Document, DocumentType
from semantic_search.models.query import Query, DateRange
from semantic_search.models.result import SearchResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Financial Semantic Search API",
        description="Production-ready semantic search for financial documents",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Global service instance
    search_service: Optional[SemanticSearchService] = None


# Pydantic models for API
if FASTAPI_AVAILABLE:
    from pydantic import BaseModel, Field
    
    class DocumentCreate(BaseModel):
        id: str = Field(..., description="Unique document identifier")
        content: str = Field(..., description="Document text content")
        entity_id: str = Field(..., description="Entity identifier")
        doc_type: str = Field(..., description="Document type")
        date: str = Field(..., description="Document date (ISO format)")
        metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class SearchRequest(BaseModel):
        text: str = Field(..., description="Search query text")
        entity_ids: Optional[List[str]] = Field(None, description="Filter by entity IDs")
        doc_types: Optional[List[str]] = Field(None, description="Filter by document types")
        date_from: Optional[str] = Field(None, description="Start date filter (ISO format)")
        date_to: Optional[str] = Field(None, description="End date filter (ISO format)")
        similarity_threshold: float = Field(0.1, ge=0.0, le=1.0, description="Minimum similarity score")
        max_results: int = Field(50, ge=1, le=1000, description="Maximum results to return")
    
    class SearchResponse(BaseModel):
        query: str
        results_count: int
        search_time: float
        results: List[Dict[str, Any]]


# API Endpoints
if FASTAPI_AVAILABLE:
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize the search service on startup."""
        global search_service
        
        try:
            search_service = await SemanticSearchService.create(
                index_path=Path("./api_index"),
                max_features=10000,
                log_level="INFO"
            )
            
            # Load sample data if available
            sample_file = Path(__file__).parent / "sample_data" / "sample_documents.json"
            if sample_file.exists():
                documents = load_sample_documents()
                if documents:
                    await search_service.add_documents(documents)
                    logger.info(f"Loaded {len(documents)} sample documents")
            
            logger.info("Search service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize search service: {e}")
            raise
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Clean up resources on shutdown."""
        global search_service
        if search_service:
            await search_service.close()
            logger.info("Search service closed")
    
    @app.get("/", summary="API Root")
    async def root():
        """API root endpoint with basic information."""
        return {
            "name": "Financial Semantic Search API",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "health": "/health",
                "search": "/search",
                "documents": "/documents",
                "stats": "/stats"
            }
        }
    
    @app.get("/health", summary="Health Check")
    async def health_check():
        """Check the health of the search service."""
        if not search_service:
            raise HTTPException(status_code=503, detail="Search service not initialized")
        
        try:
            health = await search_service.health_check()
            return health
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/stats", summary="Get Statistics")
    async def get_stats():
        """Get search service statistics."""
        if not search_service:
            raise HTTPException(status_code=503, detail="Search service not initialized")
        
        try:
            stats = await search_service.get_stats()
            return stats
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/documents", summary="Add Document", status_code=status.HTTP_201_CREATED)
    async def add_document(document: DocumentCreate, background_tasks: BackgroundTasks):
        """Add a new document to the search index."""
        if not search_service:
            raise HTTPException(status_code=503, detail="Search service not initialized")
        
        try:
            # Convert to Document object
            doc = Document(
                id=document.id,
                content=document.content,
                entity_id=document.entity_id,
                doc_type=DocumentType(document.doc_type),
                date=datetime.fromisoformat(document.date),
                metadata=document.metadata
            )
            
            # Add document in background
            background_tasks.add_task(search_service.add_document, doc)
            
            return {"message": "Document added successfully", "document_id": document.id}
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid document data: {e}")
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/search", summary="Search Documents", response_model=SearchResponse)
    async def search_documents(request: SearchRequest):
        """Search for documents matching the query."""
        if not search_service:
            raise HTTPException(status_code=503, detail="Search service not initialized")
        
        try:
            # Build date range filter
            date_range = None
            if request.date_from or request.date_to:
                start_date = datetime.fromisoformat(request.date_from) if request.date_from else None
                end_date = datetime.fromisoformat(request.date_to) if request.date_to else None
                date_range = DateRange(start=start_date, end=end_date)
            
            # Build document type filter
            doc_types = None
            if request.doc_types:
                doc_types = {DocumentType(dt) for dt in request.doc_types}
            
            # Create query
            query = Query(
                text=request.text,
                entity_ids=set(request.entity_ids) if request.entity_ids else None,
                doc_types=doc_types,
                date_range=date_range,
                similarity_threshold=request.similarity_threshold,
                max_results=request.max_results
            )
            
            # Perform search
            start_time = asyncio.get_event_loop().time()
            results = await search_service.search(query)
            search_time = asyncio.get_event_loop().time() - start_time
            
            # Convert results to JSON
            results_data = [result.to_dict() for result in results]
            
            return SearchResponse(
                query=request.text,
                results_count=len(results),
                search_time=search_time,
                results=results_data
            )
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid query: {e}")
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/documents/types", summary="Get Document Types")
    async def get_document_types():
        """Get available document types."""
        return {
            "document_types": [dt.value for dt in DocumentType]
        }


def load_sample_documents() -> List[Document]:
    """Load sample documents for the API."""
    sample_file = Path(__file__).parent / "sample_data" / "sample_documents.json"
    
    if not sample_file.exists():
        return []
    
    try:
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
        
    except Exception as e:
        logger.error(f"Failed to load sample documents: {e}")
        return []


def main():
    """Run the API server."""
    if not FASTAPI_AVAILABLE:
        print("❌ FastAPI not available. Install with: pip install fastapi uvicorn")
        return
    
    print("🚀 Starting Financial Semantic Search API Server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Search endpoint: POST http://localhost:8000/search")
    print("⏹️  Press Ctrl+C to stop")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    if FASTAPI_AVAILABLE:
        main()
    else:
        print("❌ FastAPI not available.")
        print("This is an optional example for production API deployment.")
        print("Install with: pip install fastapi uvicorn")
        print("The core semantic search functionality works without FastAPI.")
