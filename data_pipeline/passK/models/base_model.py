"""
Base model class for VLM inference.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    max_new_tokens: int = 2048
    repetition_penalty: float = 1.0
    do_sample: bool = True
    presence_penalty: float = 1.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample,
            "presence_penalty": self.presence_penalty,
        }


@dataclass 
class ModelConfig:
    """Configuration for model initialization."""
    model_path: str
    tensor_parallel_size: int = 8
    max_model_len: Optional[int] = None
    max_image_num: int = 4
    trust_remote_code: bool = True
    enforce_eager: bool = True
    dtype: str = "auto"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "max_image_num": self.max_image_num,
            "trust_remote_code": self.trust_remote_code,
            "enforce_eager": self.enforce_eager,
            "dtype": self.dtype,
        }


class BaseModel(ABC):
    """
    Abstract base class for VLM models.
    
    All model implementations should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, model_config: ModelConfig, generation_config: GenerationConfig):
        """
        Initialize the model.
        
        Args:
            model_config: Configuration for model initialization
            generation_config: Configuration for text generation
        """
        self.model_config = model_config
        self.generation_config = generation_config
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        images: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a single response for the given prompt.
        
        Args:
            prompt: The input prompt/question
            images: Optional list of image paths
            system_prompt: Optional system prompt
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def generate_batch(
        self,
        prompts: List[str],
        images_list: Optional[List[Optional[List[str]]]] = None,
        system_prompts: Optional[List[Optional[str]]] = None,
    ) -> List[str]:
        """
        Generate responses for a batch of prompts.
        
        Args:
            prompts: List of input prompts
            images_list: Optional list of image lists (one per prompt)
            system_prompts: Optional list of system prompts
            
        Returns:
            List of generated text responses
        """
        pass
    
    def generate_k_times(
        self,
        prompt: str,
        k: int,
        images: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        """
        Generate K responses for the same prompt (for pass@K evaluation).
        
        This method generates K independent samples for the same input,
        which is used for computing pass@K metrics.
        
        Args:
            prompt: The input prompt/question
            k: Number of times to generate
            images: Optional list of image paths
            system_prompt: Optional system prompt
            
        Returns:
            List of K generated text responses
        """
        # Default implementation: batch generation with same prompt K times
        prompts = [prompt] * k
        images_list = [images] * k if images else None
        system_prompts = [system_prompt] * k if system_prompt else None
        
        return self.generate_batch(prompts, images_list, system_prompts)
    
    def update_generation_config(self, **kwargs) -> None:
        """
        Update generation configuration parameters.
        
        Args:
            **kwargs: Key-value pairs to update in generation config
        """
        for key, value in kwargs.items():
            if hasattr(self.generation_config, key):
                setattr(self.generation_config, key, value)
            else:
                raise ValueError(f"Unknown generation config parameter: {key}")
    
    @property
    def model_name(self) -> str:
        """Return the model name/path."""
        return self.model_config.model_path
