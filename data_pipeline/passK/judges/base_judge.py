"""
Base judge class for answer evaluation.
"""

from abc import ABC, abstractmethod


class BaseJudge(ABC):
    """
    Abstract base class for answer judges.
    
    Judges are responsible for comparing predicted answers with
    ground truth answers and determining correctness.
    """
    
    def __init__(self):
        """Initialize the judge."""
        pass
    
    @abstractmethod
    def judge(self, prediction: str, ground_truth: str) -> bool:
        """
        Judge whether the prediction matches the ground truth.
        
        Args:
            prediction: The predicted/extracted answer
            ground_truth: The ground truth answer
            
        Returns:
            True if the prediction is correct, False otherwise
        """
        pass
    
    def __call__(self, prediction: str, ground_truth: str) -> bool:
        """
        Make the judge callable.
        
        Args:
            prediction: The predicted/extracted answer
            ground_truth: The ground truth answer
            
        Returns:
            True if the prediction is correct, False otherwise
        """
        return self.judge(prediction, ground_truth)
