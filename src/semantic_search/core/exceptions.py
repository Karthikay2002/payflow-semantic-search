"""Custom exceptions for semantic search system."""


class SemanticSearchError(Exception):
    """Base exception for semantic search operations."""
    pass


class DocumentProcessingError(SemanticSearchError):
    """Exception raised during document processing."""
    pass


class IndexError(SemanticSearchError):
    """Exception raised during index operations."""
    pass


class SearchError(SemanticSearchError):
    """Exception raised during search operations."""
    pass


class ValidationError(SemanticSearchError):
    """Exception raised during input validation."""
    pass


class StorageError(SemanticSearchError):
    """Exception raised during storage operations."""
    pass


class ConfigurationError(SemanticSearchError):
    """Exception raised for configuration issues."""
    pass
