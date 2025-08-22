"""Logging configuration for semantic search system."""

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    include_timestamp: bool = True
) -> None:
    """
    Set up structured logging for the semantic search system.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string
        include_timestamp: Whether to include timestamps
    """
    if format_string is None:
        if include_timestamp:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            format_string = "%(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        stream=sys.stdout,
        force=True
    )
    
    # Set specific loggers
    logging.getLogger("semantic_search").setLevel(getattr(logging, level.upper()))
    
    # Reduce noise from external libraries
    logging.getLogger("sklearn").setLevel(logging.WARNING)
    logging.getLogger("numpy").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {level}")


class StructuredLogger:
    """Structured logger with context support."""
    
    def __init__(self, name: str):
        """Initialize structured logger."""
        self.logger = logging.getLogger(name)
        self.context = {}
    
    def with_context(self, **kwargs) -> 'StructuredLogger':
        """Add context to logger."""
        new_logger = StructuredLogger(self.logger.name)
        new_logger.context = {**self.context, **kwargs}
        return new_logger
    
    def _format_message(self, message: str) -> str:
        """Format message with context."""
        if not self.context:
            return message
        
        context_str = " ".join([f"{k}={v}" for k, v in self.context.items()])
        return f"{message} [{context_str}]"
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(self._format_message(message))
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(self._format_message(message))
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(self._format_message(message))
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(self._format_message(message))
    
    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(self._format_message(message))
