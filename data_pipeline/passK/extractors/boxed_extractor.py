"""
Boxed answer extractor using mathruler.

Extracts answers from \boxed{} format and handles multiple-choice options.
"""

import re
from typing import Optional
from .base_extractor import BaseExtractor
from .extractor_registry import ExtractorRegistry


# Pattern to match single letter options (A-I)
CHOICE_PATTERN = re.compile(r'^([A-I])$', re.IGNORECASE)

# Pattern to extract choice from longer text (e.g., "A: xxx" or "The answer is A")
CHOICE_IN_TEXT_PATTERN = re.compile(
    r'(?:^|\b)([A-I])(?:\s*[:\.\)]|\s*$)',
    re.IGNORECASE
)

# Pattern for "answer is X" or "option X"
ANSWER_IS_PATTERN = re.compile(
    r'(?:answer\s+is|option|choice)\s*[:\s]*([A-I])\b',
    re.IGNORECASE
)


@ExtractorRegistry.register("boxed")
@ExtractorRegistry.register("mathruler")
@ExtractorRegistry.register("default")
class BoxedExtractor(BaseExtractor):
    """
    Extractor that uses mathruler's extract_boxed_content function.
    
    This extractor:
    1. First extracts content from \boxed{} using mathruler
    2. If the extracted content is a single letter (A-I), returns it as choice
    3. If the content contains a choice pattern, extracts the choice
    4. Otherwise returns the full extracted content
    """
    
    def __init__(self, extract_choice: bool = True):
        """
        Initialize the boxed extractor.
        
        Args:
            extract_choice: Whether to extract single-letter choices from the answer
        """
        super().__init__()
        self.extract_choice = extract_choice
    
    def extract(self, response: str) -> str:
        """
        Extract the answer from a model response.
        
        Uses mathruler's extract_boxed_content to get content from \boxed{},
        then optionally extracts choice letters.
        
        Args:
            response: The raw model response text
            
        Returns:
            Extracted answer string
        """
        from mathruler.grader import extract_boxed_content
        
        # First, extract content from \boxed{}
        answer = extract_boxed_content(response)
        
        if answer is None:
            answer = ""
        
        # Clean up the answer
        answer = answer.strip()
        
        # If extract_choice is enabled, try to extract choice letter
        if self.extract_choice and answer:
            choice = self._extract_choice(answer)
            if choice:
                return choice.upper()
        
        return answer
    
    def _extract_choice(self, answer: str) -> Optional[str]:
        """
        Try to extract a single-letter choice (A-I) from the answer.
        
        Args:
            answer: The extracted answer text
            
        Returns:
            Single letter choice if found, None otherwise
        """
        # Case 1: Answer is exactly a single letter
        match = CHOICE_PATTERN.match(answer.strip())
        if match:
            return match.group(1)
        
        # Case 2: Answer starts with a letter followed by separator (e.g., "A: xxx", "A. xxx")
        match = CHOICE_IN_TEXT_PATTERN.match(answer.strip())
        if match:
            return match.group(1)
        
        # Case 3: "answer is X" or "option X" pattern
        match = ANSWER_IS_PATTERN.search(answer)
        if match:
            return match.group(1)
        
        # Case 4: If answer is short and contains exactly one letter A-I
        if len(answer) <= 20:  # Only for short answers
            letters = re.findall(r'\b([A-I])\b', answer, re.IGNORECASE)
            if len(letters) == 1:
                return letters[0]
        
        return None


@ExtractorRegistry.register("raw")
class RawExtractor(BaseExtractor):
    """
    Simple extractor that returns the response as-is (with basic cleaning).
    
    Useful for cases where no special extraction is needed.
    """
    
    def __init__(self):
        super().__init__()
    
    def extract(self, response: str) -> str:
        """
        Return the response with basic cleaning.
        
        Args:
            response: The raw model response text
            
        Returns:
            Cleaned response string
        """
        return response.strip()


@ExtractorRegistry.register("choice_only")
class ChoiceOnlyExtractor(BaseExtractor):
    """
    Extractor that only looks for choice letters (A-I) in the response.
    
    Useful for multiple-choice questions where the model should output
    just a letter.
    """
    
    def __init__(self):
        super().__init__()
    
    def extract(self, response: str) -> str:
        """
        Extract a choice letter from the response.
        
        Args:
            response: The raw model response text
            
        Returns:
            Single letter choice, or empty string if not found
        """
        response = response.strip()
        
        # Try exact match first
        match = CHOICE_PATTERN.match(response)
        if match:
            return match.group(1).upper()
        
        # Try "answer is X" pattern
        match = ANSWER_IS_PATTERN.search(response)
        if match:
            return match.group(1).upper()
        
        # Try to find choice at the beginning
        match = CHOICE_IN_TEXT_PATTERN.match(response)
        if match:
            return match.group(1).upper()
        
        # Last resort: find any single letter A-I
        letters = re.findall(r'\b([A-I])\b', response, re.IGNORECASE)
        if len(letters) >= 1:
            # Return the last one (often the final answer)
            return letters[-1].upper()
        
        return ""
