# `evaluation/` — Reproducing the paper's benchmark numbers

The paper's evaluation pipeline runs on top of
[VLMEvalKit](https://github.com/open-compass/VLMEvalKit) with two
project-specific additions:

1. **AWS Bedrock Claude judge** — paper uses `claude-haiku-4.5` as the LLM
   judge (`bedrock-claude-haiku-4.5` alias).
2. **VLM model registrations** — released checkpoints are exposed to
   VLMEvalKit as `LMDeployAPI` / `LMDeployReasoningAPI` entries pointing at
   on-host vLLM (Qwen family) or LMDeploy (InternVL family) servers.

This directory ships those additions as a small set of patches that you drop
into a fresh VLMEvalKit clone — we deliberately do not vendor VLMEvalKit.

## Layout

```
evaluation/
├── README.md                            ← this file
├── run_eval.sh                          ← one-shot launcher
├── configs/
│   └── models.py                        ← model alias → port + system prompt
├── serve_examples/                      ← bring up each checkpoint server
│   ├── serve_qwen3_vl_8b_staged.sh      vllm serve …
│   ├── serve_qwen2_5_vl_7b_staged.sh
│   ├── serve_internvl3_8b_staged.sh     lmdeploy serve api_server …
│   └── serve_internvl3_5_8b_staged.sh
└── vlmevalkit_patches/                  ← apply once to your VLMEvalKit
    ├── apply_patches.py                 ← idempotent patcher (recommended)
    ├── api/bedrock_claude.py            ← BedrockClaude wrapper
    └── patch_snippets/                  ← if you'd rather patch by hand
        ├── api_init.snippet.py
        └── judge_util.snippet.py
```

## How to install

```bash
# 0) Clone VLMEvalKit alongside this repo
git clone https://github.com/open-compass/VLMEvalKit.git ../VLMEvalKit
pip install -e ../VLMEvalKit

# 1) Apply the Bedrock judge patch (idempotent — safe to re-run)
python evaluation/vlmevalkit_patches/apply_patches.py ../VLMEvalKit

# 2) Register our model aliases — append the dict from configs/models.py
#    into ../VLMEvalKit/vlmeval/config.py (look for `supported_VLM` /
#    `api_models` near the bottom of that file).

# 3) Configure AWS credentials for Bedrock
aws configure        # or export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
```

After step 1 you should be able to import the wrapper:

```bash
python -c 'from vlmeval.api import BedrockClaude; print(BedrockClaude)'
```

## How to run

The launcher takes any number of model aliases registered in
`configs/models.py`:

```bash
export VLMEVALKIT_HOME=$(realpath ../VLMEvalKit)
bash evaluation/run_eval.sh Qwen3_VL_8B_Staged Qwen3_VL_8B_Merged
```

By default it runs the **Table 1 main suite**. Set `VLMCC_BENCH` to switch:

| `VLMCC_BENCH=` | Benchmarks |
|---|---|
| `main` *(default)* | MathVista_MINI · MathVision_MINI · MathVerse_MINI_Vision_Intensive · WeMath · A-OKVQA · RealWorldQA · MMStar · POPE |
| `extended` | MathVerse_MINI_Vision_Only · DynaMath · HallusionBench · BLINK · VisOnlyQA-Synthetic · VisOnlyQA-Real · CV-Bench-2D · VStarBench · ChartQA_TEST · TextVQA_VAL |
| `all` | union of the above |
| any space-separated list | custom override |

Other knobs:

- `VLMCC_JUDGE` (default `bedrock-claude-haiku-4.5`)
- `VLMCC_NPROC` (default `8`) — judge-API parallelism
- `VLMCC_EXTRA_ARGS` — passed straight to `python run.py …`

## Bringing up the model servers

Each model alias in `configs/models.py` expects a server on a specific port
(see the `PORT` dict). Boot the right server before evaluation, e.g.:

```bash
# vLLM for Qwen
bash evaluation/serve_examples/serve_qwen3_vl_8b_staged.sh

# LMDeploy for InternVL
bash evaluation/serve_examples/serve_internvl3_8b_staged.sh
```

Each helper accepts ``MODEL=`` (HF id or local path), ``PORT=``, ``TP=``,
and ``CUDA_VISIBLE_DEVICES=`` overrides.

## Unified system prompt

Every Staged and Merged release uses one prompt at evaluation time — the
exact prompt the model was trained against (see
`training/format_prompts/math.jinja`):

```
You FIRST think about the reasoning process as an internal monologue and
then provide the final answer. The reasoning process MUST BE enclosed
within <think> </think> tags. The final answer MUST BE put in \boxed{}.
i.e. <think> reasoning here </think> \boxed{final answer here}
```

Base models are evaluated **without** a system prompt, matching the paper.

The prompt is wired in via `LMDeployReasoningAPI(system_prompt=...)`.

## Bedrock judge details

We hardcode two judge aliases in `vlmevalkit_patches/apply_patches.py`:

| alias | Bedrock model id |
|---|---|
| `bedrock-claude-haiku-4.5` | `global.anthropic.claude-haiku-4-5-20251001-v1:0` |
| `bedrock-claude-sonnet-4.5` | `global.anthropic.claude-sonnet-4-5-20250929-v1:0` |

Authentication: standard AWS chain — IAM role, `aws configure`, env vars,
or the optional `BEDROCK_API_KEY` env var for Bedrock proxy services
(see `vlmevalkit_patches/api/bedrock_claude.py`).
