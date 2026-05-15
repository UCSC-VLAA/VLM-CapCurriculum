# Snippet to insert into vlmeval/api/__init__.py
#
# Two minimal edits:
#   (1) add the import below the existing API imports
#   (2) extend __all__ to include 'BedrockClaude'

# ── (1) insert near the top of vlmeval/api/__init__.py:
from .bedrock_claude import BedrockClaude

# ── (2) inside the __all__ list, append:
__all__ = [
    # ... existing entries ...
    'BedrockClaude',
]
