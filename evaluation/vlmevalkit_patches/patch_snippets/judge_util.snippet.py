# Snippet to insert into vlmeval/dataset/utils/judge_util.py
#
# Two edits:
#   (1) Import BedrockClaude alongside the other API wrappers
#   (2) Register two model aliases and route them to BedrockClaude
#
# This patch does NOT remove or alter any existing judge alias.

def build_judge(**kwargs):
    # (1) extend the existing API import
    from ...api import (
        OpenAIWrapper, SiliconFlowAPI, HFChatModel,
        BedrockClaude,                              # ← new
    )

    model = kwargs.pop('model', None)
    kwargs.pop('nproc', None)

    # ... existing model_map = {...} block ...
    model_map = {
        # (2) add these two aliases inside the existing model_map dict:
        'bedrock-claude-haiku-4.5':  'global.anthropic.claude-haiku-4-5-20251001-v1:0',
        'bedrock-claude-sonnet-4.5': 'global.anthropic.claude-sonnet-4-5-20250929-v1:0',
        # ... existing aliases preserved ...
    }

    model_version = model_map.get(model, model)

    # (3) add this branch BEFORE the existing dispatch chain
    #     (it must come first because it short-circuits the OpenAI fallback):
    if model and model.startswith('bedrock-claude'):
        return BedrockClaude(model=model_version, **kwargs)

    # ... existing dispatch (qwen-7b / qwen-72b / deepseek / llama31-8b / OpenAIWrapper) ...
