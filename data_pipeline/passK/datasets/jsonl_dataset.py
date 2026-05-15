"""
JSONL dataset implementations for text-only QA and VQA.

Supports the following formats:
- Text QA: {"index": str, "problem": str, "answer": str, "images": [], "source": str}
- VQA: {"index": str, "problem": str, "answer": str, "images": [str, ...], "source": str}
"""

import json
import os
from typing import List, Dict, Any, Optional
from .base_dataset import BaseDataset, DataSample, DEFAULT_SYSTEM_PROMPT
from .dataset_registry import DatasetRegistry


@DatasetRegistry.register("text_qa")
@DatasetRegistry.register("textqa")
@DatasetRegistry.register("text-qa")
class TextQADataset(BaseDataset):
    """
    Dataset for text-only QA tasks.
    
    Compatible with JSONL format:
    {"index": "xxx", "problem": "...", "answer": "...", "images": [], "source": "..."}
    """
    
    def __init__(
        self,
        data_path: str,
        system_prompt_template: str = DEFAULT_SYSTEM_PROMPT,
        use_thinking: bool = True,
    ):
        """
        Initialize text QA dataset.
        
        Args:
            data_path: Path to the JSONL file
            system_prompt_template: Template for system prompt
            use_thinking: Whether to use thinking-style system prompt
        """
        super().__init__(data_path, system_prompt_template, use_thinking)
    
    def load(self) -> None:
        """Load the dataset from JSONL file."""
        self._samples = []
        
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    sample = self._parse_sample(data)
                    self._samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
                    continue
        
        print(f"Loaded {len(self._samples)} samples from {self.data_path}")
    
    def _parse_sample(self, data: Dict[str, Any]) -> DataSample:
        """
        Parse a single sample from JSON data.
        
        Args:
            data: Dictionary from JSON line
            
        Returns:
            DataSample instance
        """
        # Extract required fields
        index = str(data.get("index", ""))
        problem = data.get("problem", "")
        answer = str(data.get("answer", ""))
        source = data.get("source", "")
        
        # Images should be empty for text QA
        images = []
        
        # Store any additional fields as metadata
        known_fields = {"index", "problem", "answer", "images", "source"}
        metadata = {k: v for k, v in data.items() if k not in known_fields}
        
        return DataSample(
            index=index,
            problem=problem,
            answer=answer,
            images=images,
            source=source,
            metadata=metadata,
        )


@DatasetRegistry.register("vqa")
@DatasetRegistry.register("visual_qa")
@DatasetRegistry.register("visual-qa")
class VQADataset(BaseDataset):
    """
    Dataset for Visual Question Answering tasks.
    
    Compatible with JSONL format:
    {"index": "xxx", "problem": "...", "answer": "...", "images": ["path/to/img.jpg"], "source": "..."}
    
    The image paths in the dataset are relative paths. The image_dir parameter
    specifies the base directory to prepend to these paths.
    """
    
    def __init__(
        self,
        data_path: str,
        image_dir: str,
        system_prompt_template: str = DEFAULT_SYSTEM_PROMPT,
        use_thinking: bool = True,
    ):
        """
        Initialize VQA dataset.
        
        Args:
            data_path: Path to the JSONL file
            image_dir: Base directory for images (prepended to relative paths)
            system_prompt_template: Template for system prompt
            use_thinking: Whether to use thinking-style system prompt
        """
        super().__init__(data_path, system_prompt_template, use_thinking)
        self.image_dir = image_dir
    
    def load(self) -> None:
        """Load the dataset from JSONL file."""
        self._samples = []
        
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    sample = self._parse_sample(data)
                    self._samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
                    continue
        
        print(f"Loaded {len(self._samples)} VQA samples from {self.data_path}")
    
    def _parse_sample(self, data: Dict[str, Any]) -> DataSample:
        """
        Parse a single sample from JSON data.
        
        Args:
            data: Dictionary from JSON line
            
        Returns:
            DataSample instance
        """
        # Extract required fields
        index = str(data.get("index", ""))
        problem = data.get("problem", "")
        answer = str(data.get("answer", ""))
        source = data.get("source", "")
        
        # Process image paths - prepend image_dir to relative paths
        raw_images = data.get("images", [])
        images = []
        for img_path in raw_images:
            if img_path:
                # Convert relative path to absolute path
                full_path = os.path.join(self.image_dir, img_path)
                images.append(full_path)
        
        # Store any additional fields as metadata
        known_fields = {"index", "problem", "answer", "images", "source"}
        metadata = {k: v for k, v in data.items() if k not in known_fields}
        
        return DataSample(
            index=index,
            problem=problem,
            answer=answer,
            images=images,
            source=source,
            metadata=metadata,
        )
    
    def get_prompt(self, sample: DataSample) -> str:
        """
        Get the prompt for a VQA sample.
        
        For VQA, the problem text often contains <image> placeholder.
        We return the problem as-is since the model will handle image tokens.
        
        Args:
            sample: The data sample
            
        Returns:
            Formatted prompt string
        """
        return sample.problem
