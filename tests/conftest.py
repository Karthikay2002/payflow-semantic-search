"""Pytest configuration and shared fixtures."""

import asyncio
import tempfile
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from semantic_search.models.document import Document, DocumentType
from semantic_search.models.query import Query, DateRange
from semantic_search.api.service import SemanticSearchService


@pytest.fixture
def sample_documents() -> List[Document]:
    """Create sample financial documents for testing."""
    base_date = datetime.now() - timedelta(days=30)
    
    return [
        Document(
            id="inv_001",
            content="Invoice #INV-2024-001 from ACME Corp. Total amount: $1,250.00. "
                   "Services: Web development and consulting. Due date: 2024-01-15. "
                   "Payment terms: Net 30 days.",
            entity_id="acme_corp",
            doc_type=DocumentType.INVOICE,
            date=base_date,
            metadata={"amount": 1250.00, "currency": "USD"}
        ),
        Document(
            id="po_001",
            content="Purchase Order #PO-2024-001 for office supplies. "
                   "Vendor: Office Depot. Total: $350.75. Items: Paper, pens, folders. "
                   "Delivery date: 2024-01-10.",
            entity_id="office_depot",
            doc_type=DocumentType.PURCHASE_ORDER,
            date=base_date + timedelta(days=1),
            metadata={"amount": 350.75, "items_count": 3}
        ),
        Document(
            id="inv_002",
            content="Invoice #INV-2024-002 from TechCorp Solutions. "
                   "Software licensing fees: $2,500.00. Annual subscription. "
                   "License period: 2024-2025.",
            entity_id="techcorp",
            doc_type=DocumentType.INVOICE,
            date=base_date + timedelta(days=5),
            metadata={"amount": 2500.00, "subscription": True}
        ),
        Document(
            id="contract_001",
            content="Service Agreement with CloudHost Inc. "
                   "Monthly hosting fees: $199.99. Terms: 12 months. "
                   "Services include: server hosting, backup, support.",
            entity_id="cloudhost",
            doc_type=DocumentType.CONTRACT,
            date=base_date + timedelta(days=10),
            metadata={"monthly_fee": 199.99, "duration_months": 12}
        ),
        Document(
            id="receipt_001",
            content="Receipt from Business Travel. Flight tickets: $850.00. "
                   "Destination: New York. Business purpose: Client meeting.",
            entity_id="travel_agency",
            doc_type=DocumentType.RECEIPT,
            date=base_date + timedelta(days=15),
            metadata={"category": "travel", "amount": 850.00}
        )
    ]


@pytest.fixture
def sample_query() -> Query:
    """Create a sample search query."""
    return Query(
        text="invoice software licensing",
        similarity_threshold=0.1,
        max_results=10
    )


@pytest.fixture
def temp_index_path():
    """Create temporary directory for index storage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
async def search_service(temp_index_path):
    """Create and initialize a search service for testing."""
    async with SemanticSearchService.create(
        index_path=temp_index_path,
        max_features=1000,  # Smaller for faster tests
        log_level="WARNING"  # Reduce test output
    ) as service:
        yield service


@pytest.fixture
async def populated_service(search_service, sample_documents):
    """Create a search service with sample documents."""
    await search_service.add_documents(sample_documents)
    return search_service


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
