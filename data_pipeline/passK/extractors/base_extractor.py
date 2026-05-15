"""
Base extractor class for answer extraction.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseExtractor(ABC):
    """
    Abstract base class for answer extractors.
    
    Extractors are responsible for parsing model responses and
    extracting the final answer.
    """
    
    def __init__(self):
        """Initialize the extractor."""
        pass
    
    @abstractmethod
    def extract(self, response: str) -> str:
        """
        Extract the answer from a model response.
        
        Args:
            response: The raw model response text
            
        Returns:
            Extracted answer string
        """
        pass
    
    def __call__(self, response: str) -> str:
        """
        Make the extractor callable.
        
        Args:
            response: The raw model response text
            
        Returns:
            Extracted answer string
        """
        return self.extract(response)
