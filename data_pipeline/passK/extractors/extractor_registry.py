"""
Extractor registry for dynamic extractor loading.
"""

from typing import Dict, Type, Optional, Callable
from .base_extractor import BaseExtractor


class ExtractorRegistry:
    """
    Registry for answer extractor classes.
    """
    
    _registry: Dict[str, Type[BaseExtractor]] = {}
    
    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Decorator to register an extractor class.
        
        Args:
            name: The name to register the extractor under
            
        Returns:
            Decorator function
        """
        def decorator(extractor_class: Type[BaseExtractor]) -> Type[BaseExtractor]:
            cls._registry[name.lower()] = extractor_class
            return extractor_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseExtractor]]:
        """
        Get a registered extractor class by name.
        
        Args:
            name: The registered extractor name
            
        Returns:
            The extractor class if found, None otherwise
        """
        return cls._registry.get(name.lower())
    
    @classmethod
    def list_extractors(cls) -> list:
        """
        List all registered extractor names.
        
        Returns:
            List of registered extractor names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseExtractor:
        """
        Create an extractor instance by name.
        
        Args:
            name: The registered extractor name
            **kwargs: Arguments to pass to the extractor constructor
            
        Returns:
            Extractor instance
            
        Raises:
            ValueError: If extractor name is not registered
        """
        extractor_class = cls.get(name)
        if extractor_class is None:
            available = cls.list_extractors()
            raise ValueError(
                f"Extractor '{name}' not found. Available extractors: {available}"
            )
        return extractor_class(**kwargs)


def get_extractor(extractor_name: str = "boxed", **kwargs) -> BaseExtractor:
    """
    Convenience function to create an extractor instance.
    
    Args:
        extractor_name: Name of the extractor type (default: "boxed")
        **kwargs: Additional arguments passed to extractor constructor
        
    Returns:
        Initialized extractor instance
    """
    return ExtractorRegistry.create(extractor_name, **kwargs)
