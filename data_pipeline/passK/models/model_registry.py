"""
Model registry for dynamic model loading.
"""

from typing import Dict, Type, Optional, Callable
from .base_model import BaseModel, ModelConfig, GenerationConfig


class ModelRegistry:
    """
    Registry for VLM model classes.
    
    Supports registering models by name and creating instances dynamically.
    """
    
    _registry: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Decorator to register a model class.
        
        Args:
            name: The name to register the model under
            
        Returns:
            Decorator function
            
        Example:
            @ModelRegistry.register("qwen2.5-vl")
            class Qwen25VLModel(BaseModel):
                ...
        """
        def decorator(model_class: Type[BaseModel]) -> Type[BaseModel]:
            cls._registry[name.lower()] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseModel]]:
        """
        Get a registered model class by name.
        
        Args:
            name: The registered model name
            
        Returns:
            The model class if found, None otherwise
        """
        return cls._registry.get(name.lower())
    
    @classmethod
    def list_models(cls) -> list:
        """
        List all registered model names.
        
        Returns:
            List of registered model names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def create(
        cls,
        name: str,
        model_config: ModelConfig,
        generation_config: GenerationConfig,
    ) -> BaseModel:
        """
        Create a model instance by name.
        
        Args:
            name: The registered model name
            model_config: Configuration for model initialization
            generation_config: Configuration for text generation
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model name is not registered
        """
        model_class = cls.get(name)
        if model_class is None:
            available = cls.list_models()
            raise ValueError(
                f"Model '{name}' not found. Available models: {available}"
            )
        return model_class(model_config, generation_config)


def get_model(
    model_name: str,
    model_path: str,
    tensor_parallel_size: int = 1,
    max_model_len: Optional[int] = None,
    max_image_num: int = 4,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 512,
    repetition_penalty: float = 1.0,
    **kwargs,
) -> BaseModel:
    """
    Convenience function to create a model instance.
    
    This function provides a simpler interface for creating models
    without manually constructing config objects.
    
    Args:
        model_name: Name of the model type (e.g., "qwen2.5-vl", "qwen3-vl")
        model_path: Path to the model (local or HuggingFace)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        max_model_len: Maximum model context length
        max_image_num: Maximum number of images per prompt
        temperature: Sampling temperature
        top_p: Top-p (nucleus) sampling parameter
        max_new_tokens: Maximum new tokens to generate
        repetition_penalty: Repetition penalty
        **kwargs: Additional arguments passed to model config
        
    Returns:
        Initialized model instance
    """
    model_config = ModelConfig(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        max_image_num=max_image_num,
        **{k: v for k, v in kwargs.items() if hasattr(ModelConfig, k)},
    )
    
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
    )
    
    return ModelRegistry.create(model_name, model_config, generation_config)
