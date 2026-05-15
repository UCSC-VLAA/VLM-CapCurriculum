"""
Qwen VL model implementations (Qwen2.5-VL and Qwen3-VL).

Supports both vLLM backend for efficient inference.
"""

import os
from typing import List, Dict, Any, Optional, Union
from .base_model import BaseModel, ModelConfig, GenerationConfig
from .model_registry import ModelRegistry


class QwenVLModel(BaseModel):
    """
    Base class for Qwen VL models using vLLM backend.
    
    This class provides common functionality for Qwen2.5-VL and Qwen3-VL models.
    """
    
    def __init__(self, model_config: ModelConfig, generation_config: GenerationConfig):
        super().__init__(model_config, generation_config)
        self._init_model()
        self._init_sampling_params()
    
    def _init_model(self) -> None:
        """Initialize the vLLM model and processor."""
        from vllm import LLM
        from transformers import AutoProcessor
        
        # Build vLLM kwargs
        vllm_kwargs = {
            "model": self.model_config.model_path,
            "tensor_parallel_size": self.model_config.tensor_parallel_size,
            "enforce_eager": self.model_config.enforce_eager,
            "trust_remote_code": self.model_config.trust_remote_code,
            "limit_mm_per_prompt": {"image": self.model_config.max_image_num},
            "disable_log_stats": True,  # Disable vLLM stats logging
        }
        
        if self.model_config.max_model_len is not None:
            vllm_kwargs["max_model_len"] = self.model_config.max_model_len
        
        if self.model_config.dtype != "auto":
            vllm_kwargs["dtype"] = self.model_config.dtype
        
        self.llm = LLM(**vllm_kwargs)
        self.processor = AutoProcessor.from_pretrained(
            self.model_config.model_path,
            trust_remote_code=self.model_config.trust_remote_code,
        )
    
    def _init_sampling_params(self) -> None:
        """Initialize vLLM sampling parameters."""
        from vllm import SamplingParams
        
        self.sampling_params = SamplingParams(
            temperature=self.generation_config.temperature,
            top_p=self.generation_config.top_p,
            top_k=self.generation_config.top_k,
            max_tokens=self.generation_config.max_new_tokens,
            repetition_penalty=self.generation_config.repetition_penalty,
        )
    
    def _build_messages(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build chat messages in Qwen VL format.
        
        Args:
            prompt: The user prompt
            images: Optional list of image paths
            system_prompt: Optional system prompt
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Build user message content
        content = []
        
        if images:
            for i, image_path in enumerate(images):
                if len(images) > 1:
                    content.append({"type": "text", "text": f"<image_{i+1}>: "})
                content.append({"type": "image", "image": image_path})
        
        content.append({"type": "text", "text": prompt})
        
        messages.append({
            "role": "user",
            "content": content
        })
        
        return messages
    
    def _process_messages(
        self,
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process messages into vLLM input format.
        
        Args:
            messages: Chat messages in Qwen format
            
        Returns:
            Dictionary with prompt and optional multi_modal_data
        """
        from qwen_vl_utils import process_vision_info
        
        # Apply chat template
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Process vision inputs
        image_inputs, video_inputs = process_vision_info(messages)
        
        llm_inputs = {"prompt": prompt}
        
        if image_inputs is not None:
            llm_inputs["multi_modal_data"] = {"image": image_inputs}
        
        return llm_inputs
    
    def generate(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a single response.
        
        Args:
            prompt: The input prompt
            images: Optional list of image paths
            system_prompt: Optional system prompt
            
        Returns:
            Generated text response
        """
        messages = self._build_messages(prompt, images, system_prompt)
        llm_inputs = self._process_messages(messages)
        
        outputs = self.llm.generate(
            [llm_inputs], 
            sampling_params=self.sampling_params,
            use_tqdm=False,  # Disable vLLM progress bar
        )
        return outputs[0].outputs[0].text
    
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
            images_list: Optional list of image lists
            system_prompts: Optional list of system prompts
            
        Returns:
            List of generated text responses
        """
        batch_size = len(prompts)
        
        # Handle None lists
        if images_list is None:
            images_list = [None] * batch_size
        if system_prompts is None:
            system_prompts = [None] * batch_size
        
        # Build all inputs
        llm_inputs_list = []
        for prompt, images, system_prompt in zip(prompts, images_list, system_prompts):
            messages = self._build_messages(prompt, images, system_prompt)
            llm_inputs = self._process_messages(messages)
            llm_inputs_list.append(llm_inputs)
        
        # Batch generation
        outputs = self.llm.generate(
            llm_inputs_list, 
            sampling_params=self.sampling_params,
            use_tqdm=False,  # Disable vLLM progress bar
        )
        
        return [output.outputs[0].text for output in outputs]
    
    def generate_k_times(
        self,
        prompt: str,
        k: int,
        images: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        """
        Generate K responses for the same prompt (for pass@K evaluation).
        
        This implementation batches all K generations together for efficiency.
        
        Args:
            prompt: The input prompt
            k: Number of generations
            images: Optional list of image paths
            system_prompt: Optional system prompt
            
        Returns:
            List of K generated responses
        """
        # Build messages once
        messages = self._build_messages(prompt, images, system_prompt)
        llm_inputs = self._process_messages(messages)
        
        # Create K copies of the same input
        llm_inputs_list = [llm_inputs] * k
        
        # Batch generation with same input K times
        outputs = self.llm.generate(
            llm_inputs_list, 
            sampling_params=self.sampling_params,
            use_tqdm=False,  # Disable vLLM progress bar
        )
        
        return [output.outputs[0].text for output in outputs]
    
    def update_generation_config(self, **kwargs) -> None:
        """Update generation config and reinitialize sampling params."""
        super().update_generation_config(**kwargs)
        self._init_sampling_params()


# Register Qwen2.5-VL model
@ModelRegistry.register("qwen2.5-vl")
@ModelRegistry.register("qwen2_5-vl")
@ModelRegistry.register("qwen25-vl")
class Qwen25VLModel(QwenVLModel):
    """
    Qwen2.5-VL model implementation.
    
    Supports Qwen2.5-VL series models (3B, 7B, 72B).
    """
    pass


# Register Qwen3-VL model
@ModelRegistry.register("qwen3-vl")
@ModelRegistry.register("qwen3_vl")
class Qwen3VLModel(QwenVLModel):
    """
    Qwen3-VL model implementation.
    
    Supports Qwen3-VL series models.
    Note: Qwen3-VL uses the same interface as Qwen2.5-VL.
    """
    pass


# Also register with full names for convenience
@ModelRegistry.register("qwen2-vl")
@ModelRegistry.register("qwen2_vl")
class Qwen2VLModel(QwenVLModel):
    """
    Qwen2-VL model implementation (legacy).
    
    Supports Qwen2-VL series models.
    """
    pass
