# `training/` — RLVR training on top of EasyR1

This directory contains the *paper-specific* assets — reward functions, format
prompts, base config, and per-experiment launch scripts. The training engine
itself ([EasyR1](https://github.com/hiyouga/EasyR1)) is not vendored here.

## How to launch a run

1. Clone EasyR1 separately and install it:

   ```bash
   git clone https://github.com/hiyouga/EasyR1.git ../EasyR1
   pip install -e ../EasyR1
   ```

2. Source the env header (sets `EASYR1_HOME`, `VLMCC_*` data and prompt paths,
   default backbone HF ids, etc.):

   ```bash
   source training/_env.sh
   ```

3. Run the script you want. Most stages need a `MODEL_PATH` pointing at the
   previous-stage checkpoint — examples in the script headers.

   ```bash
   # Stage 1 from base
   bash training/examples/qwen3_vl_8b/stage1_perception.sh

   # Stage 2 chained from Stage 1
   MODEL_PATH=${EASYR1_HOME}/checkpoints/${VLMCC_PROJECT}/qwen3_vl_8b_stage1_perception/global_step_96/actor/huggingface \
       bash training/examples/qwen3_vl_8b/stage2_text_reasoning.sh
   ```

The end-to-end Qwen3-VL-8B recipe is wrapped in
`scripts/quickstart_qwen3vl_8b_staged.sh` (one shot).

## Directory layout

```
training/
├── _env.sh                       sourced before each run; controls all paths
├── configs/
│   └── config.yaml               EasyR1 base config (copied from upstream)
├── format_prompts/
│   └── math.jinja                <think>...</think>\boxed{...}  ← unified prompt
├── reward_functions/
│   └── math.py                   timeout-protected math grader (used by all stages)
└── examples/
    ├── qwen3_vl_8b/              ★ primary backbone — stage{1,2,3}, merged
    ├── qwen2_5_vl_7b/            second backbone — same four scripts
    ├── internvl3_8b/             InternVL3-8B (note its damped optim knobs)
    ├── internvl3_5_8b/           InternVL3.5-8B
    ├── ablations/                stage-order, encoder freezing, SFT-vs-RL
    └── curriculum/               capability × difficulty (Sec 4.5)
```

## Unified prompt across all stages

All staged + merged + ablation runs in this repo use a **single system /
format prompt** for both training and evaluation:

```
You FIRST think about the reasoning process as an internal monologue and then
provide the final answer. The reasoning process MUST BE enclosed within
<think> </think> tags. The final answer MUST BE put in \boxed{}.
i.e. <think> reasoning here </think> \boxed{final answer here}
```

This is the prompt the released checkpoints learned to follow, so the same
prompt is used at evaluation time (see `evaluation/configs/models.py`).
Earlier internal experiments tried per-stage formats (e.g. `<description>`
tags for Stage 1, `<description><think><answer>` for Stage 3) but they were
**not** the configuration that produced the paper's reported numbers.

## Per-backbone notes

- **Qwen3-VL-8B / Qwen2.5-VL-7B** — vanilla GRPO settings, vision tower open
  in Stage 1/3, frozen in Stage 2.
- **InternVL3-8B** — Stage 2 needs damped optimisation (`lr=3e-7`,
  `kl_coef=5e-2`, `clip_ratio=0.15`, `max_grad_norm=0.5`) to avoid entropy
  explosion. Don't strip these. `max_prompt_length=4096`,
  `offload_optimizer=true` to fit memory.
- **InternVL3.5-8B** — same memory-side adjustments as InternVL3-8B but no
  entropy-explosion damping required.

## Hyperparameter consistency

All stage-aligned hyper-parameters in the scripts here mirror the runs that
produced the numbers in the paper. The key knobs (and their paper
provenance):

| Knob | Value | Source |
|---|---|---|
| `data.max_prompt_length` | 2048 (Qwen) / 4096 (InternVL) | Sec 4.1 |
| `worker.rollout.n` (group size) | 5 | Sec 4.1 |
| Stage 1 epochs | 16 | Sec 4.1 |
| Stage 2/3 epochs | 15 | Sec 4.1 |
| Stage steps (90 / 375 / 465) | implicit via dataset size × epochs | Sec 4.1 |
| Total merged steps | 930 | Sec 4.1 |

If you change any of these you are *not* reproducing the paper's results.
