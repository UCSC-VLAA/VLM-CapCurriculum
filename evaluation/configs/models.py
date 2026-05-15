"""Model registrations for VLMEvalKit.

Each released checkpoint is served as an OpenAI-compatible chat server
(via vLLM for the Qwen family, via LMDeploy for the InternVL family) and
exposed to VLMEvalKit through ``LMDeployAPI`` / ``LMDeployReasoningAPI``
entries.

To plug these into a fresh VLMEvalKit checkout, append the dictionary below
to ``vlmeval/config.py``'s ``api_models`` dict (or merge into
``supported_VLM`` directly).

Usage in a VLMEvalKit run::

    python run.py --data WeMath MathVista_MINI ... \\
        --model Qwen3_VL_8B_Staged \\
        --judge bedrock-claude-haiku-4.5

Bring up each model's backing server before launching the eval; see
``evaluation/serve_examples/`` for ready-to-run scripts.

All Staged / Merged checkpoints share **one** system prompt — the same
prompt used during training (see `training/format_prompts/math.jinja`):

  > You FIRST think about the reasoning process as an internal monologue and
  > then provide the final answer. The reasoning process MUST BE enclosed
  > within <think> </think> tags. The final answer MUST BE put in \\boxed{}.
  > i.e. <think> reasoning here </think> \\boxed{final answer here}

Base models do **not** receive a system prompt — they're evaluated under
their native chat template, matching the paper's setup.
"""
from functools import partial

from vlmeval.api.lmdeploy import LMDeployAPI, LMDeployReasoningAPI


# ---------------------------------------------------------------------------
# Unified system prompt used by every Staged / Merged model
# ---------------------------------------------------------------------------
UNIFIED_SYSTEM_PROMPT = (
    "You FIRST think about the reasoning process as an internal monologue "
    "and then provide the final answer. The reasoning process MUST BE "
    "enclosed within <think> </think> tags. The final answer MUST BE put "
    "in \\boxed{}. i.e. <think> reasoning here </think> "
    "\\boxed{final answer here}"
)


# ---------------------------------------------------------------------------
# Default port assignments
# ---------------------------------------------------------------------------
# Override the port for any alias by editing this map. The matching
# server-launch helper reads ports from this same dict.

PORT = {
    # base backbones (no system prompt)
    "Qwen2_5_VL_7B_Base":      23333,
    "Qwen3_VL_8B_Base":        23334,
    "InternVL3_8B_Base":       23335,
    "InternVL3_5_8B_Base":     23336,

    # our staged-training releases
    "Qwen2_5_VL_7B_Staged":    23340,
    "Qwen3_VL_8B_Staged":      23341,
    "InternVL3_8B_Staged":     23342,
    "InternVL3_5_8B_Staged":   23343,

    # merged-training baselines
    "Qwen2_5_VL_7B_Merged":    23350,
    "Qwen3_VL_8B_Merged":      23351,
    "InternVL3_8B_Merged":     23352,
    "InternVL3_5_8B_Merged":   23353,
}


def _api(port: int, *, with_system_prompt: bool):
    if with_system_prompt:
        return partial(
            LMDeployReasoningAPI,
            api_base=f"http://0.0.0.0:{port}/v1/chat/completions",
            temperature=0,
            system_prompt=UNIFIED_SYSTEM_PROMPT,
            retry=10,
        )
    return partial(
        LMDeployAPI,
        api_base=f"http://0.0.0.0:{port}/v1/chat/completions",
        temperature=0,
        retry=10,
    )


vlm_capcurriculum_models = {
    # ---- baseline backbones — native chat template, no system prompt ------
    "Qwen2_5_VL_7B_Base":      _api(PORT["Qwen2_5_VL_7B_Base"],      with_system_prompt=False),
    "Qwen3_VL_8B_Base":        _api(PORT["Qwen3_VL_8B_Base"],        with_system_prompt=False),
    "InternVL3_8B_Base":       _api(PORT["InternVL3_8B_Base"],       with_system_prompt=False),
    "InternVL3_5_8B_Base":     _api(PORT["InternVL3_5_8B_Base"],     with_system_prompt=False),

    # ---- our staged-training releases — unified system prompt ------------
    "Qwen2_5_VL_7B_Staged":    _api(PORT["Qwen2_5_VL_7B_Staged"],    with_system_prompt=True),
    "Qwen3_VL_8B_Staged":      _api(PORT["Qwen3_VL_8B_Staged"],      with_system_prompt=True),
    "InternVL3_8B_Staged":     _api(PORT["InternVL3_8B_Staged"],     with_system_prompt=True),
    "InternVL3_5_8B_Staged":   _api(PORT["InternVL3_5_8B_Staged"],   with_system_prompt=True),

    # ---- merged-training baselines — same unified system prompt ----------
    "Qwen2_5_VL_7B_Merged":    _api(PORT["Qwen2_5_VL_7B_Merged"],    with_system_prompt=True),
    "Qwen3_VL_8B_Merged":      _api(PORT["Qwen3_VL_8B_Merged"],      with_system_prompt=True),
    "InternVL3_8B_Merged":     _api(PORT["InternVL3_8B_Merged"],     with_system_prompt=True),
    "InternVL3_5_8B_Merged":   _api(PORT["InternVL3_5_8B_Merged"],   with_system_prompt=True),
}


__all__ = ["vlm_capcurriculum_models", "PORT", "UNIFIED_SYSTEM_PROMPT"]
