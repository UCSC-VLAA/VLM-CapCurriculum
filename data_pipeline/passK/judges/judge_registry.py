"""
Judge registry for dynamic judge loading.
"""

from typing import Dict, Type, Optional, Callable
from .base_judge import BaseJudge


class JudgeRegistry:
    """
    Registry for answer judge classes.
    """
    
    _registry: Dict[str, Type[BaseJudge]] = {}
    
    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Decorator to register a judge class.
        
        Args:
            name: The name to register the judge under
            
        Returns:
            Decorator function
        """
        def decorator(judge_class: Type[BaseJudge]) -> Type[BaseJudge]:
            cls._registry[name.lower()] = judge_class
            return judge_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseJudge]]:
        """
        Get a registered judge class by name.
        
        Args:
            name: The registered judge name
            
        Returns:
            The judge class if found, None otherwise
        """
        return cls._registry.get(name.lower())
    
    @classmethod
    def list_judges(cls) -> list:
        """
        List all registered judge names.
        
        Returns:
            List of registered judge names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseJudge:
        """
        Create a judge instance by name.
        
        Args:
            name: The registered judge name
            **kwargs: Arguments to pass to the judge constructor
            
        Returns:
            Judge instance
            
        Raises:
            ValueError: If judge name is not registered
        """
        judge_class = cls.get(name)
        if judge_class is None:
            available = cls.list_judges()
            raise ValueError(
                f"Judge '{name}' not found. Available judges: {available}"
            )
        return judge_class(**kwargs)


def get_judge(judge_name: str = "mathruler", **kwargs) -> BaseJudge:
    """
    Convenience function to create a judge instance.
    
    Args:
        judge_name: Name of the judge type (default: "mathruler")
        **kwargs: Additional arguments passed to judge constructor
        
    Returns:
        Initialized judge instance
    """
    return JudgeRegistry.create(judge_name, **kwargs)
