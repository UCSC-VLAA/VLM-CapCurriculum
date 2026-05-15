"""
Base dataset class for VLM inference.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


# Default system prompt from EasyR1 math.jinja template
DEFAULT_SYSTEM_PROMPT = (
    "{content} You FIRST think about the reasoning process as an internal monologue "
    "and then provide the final answer. The reasoning process MUST BE enclosed within "
    "<thinking> </thinking> tags. The final answer MUST BE put in \\boxed{{}}. "
    "i.e. <thinking> reasoning here </thinking> \\boxed{{final answer here}}"
)

# Simple system prompt without thinking tags (for non-reasoning evaluation)
SIMPLE_SYSTEM_PROMPT = "{content}"


@dataclass
class DataSample:
    """
    A single data sample for inference.
    
    Attributes:
        index: Unique identifier for the sample
        problem: The problem/question text
        answer: The ground truth answer
        images: List of image paths (empty for text-only)
        source: Data source identifier
        metadata: Additional metadata from the original sample
    """
    index: str
    problem: str
    answer: str
    images: List[str] = field(default_factory=list)
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "index": self.index,
            "problem": self.problem,
            "answer": self.answer,
            "images": self.images,
            "source": self.source,
        }
        # Include any additional metadata
        result.update(self.metadata)
        return result
    
    @property
    def has_images(self) -> bool:
        """Check if this sample has images."""
        return len(self.images) > 0


class BaseDataset(ABC):
    """
    Abstract base class for datasets.
    
    All dataset implementations should inherit from this class.
    """
    
    def __init__(
        self,
        data_path: str,
        system_prompt_template: str = DEFAULT_SYSTEM_PROMPT,
        use_thinking: bool = True,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the dataset file or directory
            system_prompt_template: Template for system prompt with {content} placeholder
            use_thinking: Whether to use thinking-style system prompt
        """
        self.data_path = data_path
        self.system_prompt_template = system_prompt_template if use_thinking else SIMPLE_SYSTEM_PROMPT
        self.use_thinking = use_thinking
        self._samples: List[DataSample] = []
        
    @abstractmethod
    def load(self) -> None:
        """Load the dataset from disk."""
        pass
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self._samples)
    
    def __getitem__(self, idx: int) -> DataSample:
        """Get a sample by index."""
        return self._samples[idx]
    
    def __iter__(self):
        """Iterate over samples."""
        return iter(self._samples)
    
    def get_system_prompt(self, content: str = "") -> str:
        """
        Get the formatted system prompt.
        
        Args:
            content: Optional content to insert into the template
            
        Returns:
            Formatted system prompt
        """
        return self.system_prompt_template.format(content=content).strip()
    
    def get_prompt(self, sample: DataSample) -> str:
        """
        Get the prompt for a sample.
        
        By default, returns the problem text. Subclasses can override
        to add additional formatting.
        
        Args:
            sample: The data sample
            
        Returns:
            Formatted prompt string
        """
        return sample.problem
    
    def get_images(self, sample: DataSample) -> Optional[List[str]]:
        """
        Get the image paths for a sample.
        
        Args:
            sample: The data sample
            
        Returns:
            List of image paths, or None if no images
        """
        if sample.has_images:
            return sample.images
        return None
    
    @property
    def samples(self) -> List[DataSample]:
        """Get all samples."""
        return self._samples
    
    @property
    def is_vqa(self) -> bool:
        """Check if this is a VQA dataset (has images)."""
        if len(self._samples) == 0:
            return False
        # Check first sample for images
        return self._samples[0].has_images
