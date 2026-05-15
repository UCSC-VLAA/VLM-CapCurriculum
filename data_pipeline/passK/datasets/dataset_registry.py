"""
Dataset registry for dynamic dataset loading.
"""

from typing import Dict, Type, Optional, Callable
from .base_dataset import BaseDataset


class DatasetRegistry:
    """
    Registry for dataset classes.
    
    Supports registering datasets by name and creating instances dynamically.
    """
    
    _registry: Dict[str, Type[BaseDataset]] = {}
    
    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Decorator to register a dataset class.
        
        Args:
            name: The name to register the dataset under
            
        Returns:
            Decorator function
        """
        def decorator(dataset_class: Type[BaseDataset]) -> Type[BaseDataset]:
            cls._registry[name.lower()] = dataset_class
            return dataset_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseDataset]]:
        """
        Get a registered dataset class by name.
        
        Args:
            name: The registered dataset name
            
        Returns:
            The dataset class if found, None otherwise
        """
        return cls._registry.get(name.lower())
    
    @classmethod
    def list_datasets(cls) -> list:
        """
        List all registered dataset names.
        
        Returns:
            List of registered dataset names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseDataset:
        """
        Create a dataset instance by name.
        
        Args:
            name: The registered dataset name
            **kwargs: Arguments to pass to the dataset constructor
            
        Returns:
            Dataset instance
            
        Raises:
            ValueError: If dataset name is not registered
        """
        dataset_class = cls.get(name)
        if dataset_class is None:
            available = cls.list_datasets()
            raise ValueError(
                f"Dataset '{name}' not found. Available datasets: {available}"
            )
        return dataset_class(**kwargs)


def get_dataset(
    dataset_name: str,
    data_path: str,
    image_dir: Optional[str] = None,
    use_thinking: bool = True,
    **kwargs,
) -> BaseDataset:
    """
    Convenience function to create a dataset instance.
    
    This function provides a simpler interface for creating datasets.
    It automatically selects the appropriate dataset type based on name
    or auto-detects from the data.
    
    Args:
        dataset_name: Name of the dataset type ("text_qa", "vqa", or "auto")
        data_path: Path to the dataset file
        image_dir: Base directory for images (required for VQA)
        use_thinking: Whether to use thinking-style system prompt
        **kwargs: Additional arguments passed to dataset constructor
        
    Returns:
        Initialized dataset instance
    """
    # Handle auto-detection
    if dataset_name.lower() == "auto":
        dataset_name = _auto_detect_dataset_type(data_path)
    
    # Build kwargs
    dataset_kwargs = {
        "data_path": data_path,
        "use_thinking": use_thinking,
        **kwargs,
    }
    
    # Add image_dir for VQA datasets
    if image_dir is not None:
        dataset_kwargs["image_dir"] = image_dir
    
    dataset = DatasetRegistry.create(dataset_name, **dataset_kwargs)
    dataset.load()
    return dataset


def _auto_detect_dataset_type(data_path: str) -> str:
    """
    Auto-detect dataset type by examining the first line.
    
    Args:
        data_path: Path to the dataset file
        
    Returns:
        Dataset type name ("text_qa" or "vqa")
    """
    import json
    
    with open(data_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if first_line:
            sample = json.loads(first_line)
            images = sample.get("images", [])
            if images and len(images) > 0:
                return "vqa"
    
    return "text_qa"
