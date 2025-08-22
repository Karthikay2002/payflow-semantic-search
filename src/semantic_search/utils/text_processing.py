"""Text processing utilities for document content."""

import re
import string
from typing import List, Set


class TextProcessor:
    """Text processing utilities for financial documents."""
    
    def __init__(self):
        """Initialize text processor with financial document patterns."""
        # Common financial terms to preserve
        self.financial_terms = {
            'invoice', 'purchase', 'order', 'payment', 'amount', 'total',
            'subtotal', 'tax', 'discount', 'quantity', 'price', 'cost',
            'vendor', 'supplier', 'customer', 'client', 'billing', 'account',
            'contract', 'agreement', 'terms', 'conditions', 'due', 'date'
        }
        
        # Patterns for financial data
        self.amount_pattern = re.compile(r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?')
        self.date_pattern = re.compile(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}')
        self.invoice_pattern = re.compile(r'(?:invoice|inv)[\s#]*(\w+)', re.IGNORECASE)
        self.po_pattern = re.compile(r'(?:po|purchase\s+order)[\s#]*(\w+)', re.IGNORECASE)
        
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text for semantic search.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Preserve important financial patterns
        text = self._preserve_financial_patterns(text)
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^\w\s$.,/-]', ' ', text)
        
        # Remove extra spaces again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _preserve_financial_patterns(self, text: str) -> str:
        """Preserve important financial patterns during cleaning."""
        # Replace currency amounts with normalized tokens
        text = self.amount_pattern.sub('AMOUNT_TOKEN', text)
        
        # Replace dates with normalized tokens
        text = self.date_pattern.sub('DATE_TOKEN', text)
        
        # Replace invoice numbers with tokens
        text = self.invoice_pattern.sub('INVOICE_TOKEN', text)
        
        # Replace PO numbers with tokens
        text = self.po_pattern.sub('PO_TOKEN', text)
        
        return text
    
    def extract_key_terms(self, text: str) -> Set[str]:
        """
        Extract key financial terms from text.
        
        Args:
            text: Document text
            
        Returns:
            Set of key terms found in text
        """
        text_lower = text.lower()
        found_terms = set()
        
        for term in self.financial_terms:
            if term in text_lower:
                found_terms.add(term)
        
        return found_terms
    
    def extract_amounts(self, text: str) -> List[str]:
        """Extract monetary amounts from text."""
        return self.amount_pattern.findall(text)
    
    def extract_dates(self, text: str) -> List[str]:
        """Extract dates from text."""
        return self.date_pattern.findall(text)
    
    def extract_invoice_numbers(self, text: str) -> List[str]:
        """Extract invoice numbers from text."""
        matches = self.invoice_pattern.findall(text)
        return [match for match in matches if match]
    
    def extract_po_numbers(self, text: str) -> List[str]:
        """Extract purchase order numbers from text."""
        matches = self.po_pattern.findall(text)
        return [match for match in matches if match]
    
    def generate_context_snippet(
        self, 
        text: str, 
        query_terms: List[str], 
        max_length: int = 200
    ) -> str:
        """
        Generate context snippet around query terms.
        
        Args:
            text: Full document text
            query_terms: Terms to find context for
            max_length: Maximum snippet length
            
        Returns:
            Context snippet with query terms highlighted
        """
        if not text or not query_terms:
            return text[:max_length] if text else ""
        
        text_lower = text.lower()
        query_lower = [term.lower() for term in query_terms]
        
        # Find the best position to extract snippet
        best_pos = 0
        max_matches = 0
        
        # Sliding window to find position with most query terms
        window_size = max_length
        for i in range(0, len(text) - window_size + 1, 20):
            window = text_lower[i:i + window_size]
            matches = sum(1 for term in query_lower if term in window)
            
            if matches > max_matches:
                max_matches = matches
                best_pos = i
        
        # Extract snippet
        snippet = text[best_pos:best_pos + max_length]
        
        # Ensure we don't cut words
        if best_pos > 0 and not snippet.startswith(' '):
            space_pos = snippet.find(' ')
            if space_pos > 0:
                snippet = snippet[space_pos + 1:]
        
        if len(snippet) == max_length and not text[best_pos + max_length:].startswith(' '):
            space_pos = snippet.rfind(' ')
            if space_pos > max_length * 0.8:  # Don't cut too much
                snippet = snippet[:space_pos]
        
        # Add ellipsis if needed
        if best_pos > 0:
            snippet = "..." + snippet
        if best_pos + len(snippet) < len(text):
            snippet = snippet + "..."
        
        return snippet.strip()
