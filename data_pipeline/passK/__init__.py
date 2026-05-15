"""
PassK: VLM Inference Pipeline for Pass Rate Evaluation.

This package provides tools for running VLM inference multiple times
and computing pass rates for QA and VQA tasks.
"""

from .models import get_model, ModelRegistry
from .datasets import get_dataset, DatasetRegistry
from .extractors import get_extractor, ExtractorRegistry
from .judges import get_judge, JudgeRegistry
from .metrics import compute_pass_rate

__all__ = [
    "get_model",
    "get_dataset", 
    "get_extractor",
    "get_judge",
    "compute_pass_rate",
    "ModelRegistry",
    "DatasetRegistry",
    "ExtractorRegistry",
    "JudgeRegistry",
]
