"""
Pass rate computation for individual samples.

For each sample, pass_rate = number of correct predictions / K.
"""

from typing import List


def compute_pass_rate(correctness: List[bool]) -> float:
    """
    Compute pass rate for a single sample.
    
    Pass rate = number of correct predictions / total predictions (K).
    
    Args:
        correctness: List of K boolean values indicating correctness of each prediction
        
    Returns:
        Pass rate as a float between 0.0 and 1.0
        
    Example:
        >>> compute_pass_rate([True, False, False, True, False])
        0.4
        >>> compute_pass_rate([True, True, True])
        1.0
        >>> compute_pass_rate([False, False, False])
        0.0
    """
    if not correctness:
        return 0.0
    return sum(correctness) / len(correctness)
