"""
MathRuler-based judge for answer evaluation.

Uses mathruler's grade_answer function for flexible answer matching.
"""

from .base_judge import BaseJudge
from .judge_registry import JudgeRegistry


@JudgeRegistry.register("mathruler")
@JudgeRegistry.register("default")
@JudgeRegistry.register("math")
class MathRulerJudge(BaseJudge):
    """
    Judge that uses mathruler's grade_answer function.
    
    This judge provides robust answer matching that handles:
    - Numerical equivalence (e.g., "1.0" == "1")
    - LaTeX expressions
    - Fraction equivalence
    - And more mathematical formats
    """
    
    def __init__(self):
        """Initialize the MathRuler judge."""
        super().__init__()
    
    def judge(self, prediction: str, ground_truth: str) -> bool:
        """
        Judge whether the prediction matches the ground truth.
        
        Uses mathruler's grade_answer for flexible matching.
        
        Args:
            prediction: The predicted/extracted answer
            ground_truth: The ground truth answer
            
        Returns:
            True if the prediction is correct, False otherwise
        """
        from mathruler.grader import grade_answer
        
        # Handle empty predictions
        if not prediction or prediction.strip() == "":
            return False
        
        # Use mathruler's grade_answer for comparison
        return grade_answer(prediction, ground_truth)


@JudgeRegistry.register("exact")
@JudgeRegistry.register("exact_match")
class ExactMatchJudge(BaseJudge):
    """
    Judge that performs exact string matching.
    
    Useful for simple cases where exact match is sufficient,
    like multiple-choice questions.
    """
    
    def __init__(self, case_sensitive: bool = False, strip: bool = True):
        """
        Initialize the exact match judge.
        
        Args:
            case_sensitive: Whether to perform case-sensitive matching
            strip: Whether to strip whitespace before comparison
        """
        super().__init__()
        self.case_sensitive = case_sensitive
        self.strip = strip
    
    def judge(self, prediction: str, ground_truth: str) -> bool:
        """
        Judge whether the prediction exactly matches the ground truth.
        
        Args:
            prediction: The predicted/extracted answer
            ground_truth: The ground truth answer
            
        Returns:
            True if exact match, False otherwise
        """
        if not prediction:
            return False
        
        pred = prediction
        gt = ground_truth
        
        if self.strip:
            pred = pred.strip()
            gt = gt.strip()
        
        if not self.case_sensitive:
            pred = pred.lower()
            gt = gt.lower()
        
        return pred == gt


@JudgeRegistry.register("choice")
@JudgeRegistry.register("multiple_choice")
class ChoiceJudge(BaseJudge):
    """
    Judge specialized for multiple-choice questions.
    
    Compares single-letter choices (A-I) in a case-insensitive manner.
    """
    
    def __init__(self):
        """Initialize the choice judge."""
        super().__init__()
    
    def judge(self, prediction: str, ground_truth: str) -> bool:
        """
        Judge whether the predicted choice matches the ground truth.
        
        Args:
            prediction: The predicted choice (e.g., "A", "a", "B")
            ground_truth: The ground truth choice
            
        Returns:
            True if choices match, False otherwise
        """
        if not prediction:
            return False
        
        # Normalize both to uppercase single letters
        pred = prediction.strip().upper()
        gt = ground_truth.strip().upper()
        
        # Handle cases where prediction might have extra text
        # Extract first letter if it's a valid choice
        if len(pred) > 1:
            import re
            match = re.search(r'([A-I])', pred, re.IGNORECASE)
            if match:
                pred = match.group(1).upper()
        
        return pred == gt
