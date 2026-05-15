# `passK/` — measure sample-level difficulty (pass rate)

The `predictions` / `correctness` / `pass_rate` columns shipped with
[`UCSC-VLAA/VLM-CapCurriculum-Perception`](https://huggingface.co/datasets/UCSC-VLAA/VLM-CapCurriculum-Perception),
[`-TextReasoning`](https://huggingface.co/datasets/UCSC-VLAA/VLM-CapCurriculum-TextReasoning), and
[`-VisualReasoning`](https://huggingface.co/datasets/UCSC-VLAA/VLM-CapCurriculum-VisualReasoning)
are produced by this small toolkit.

For each sample, we sample **K** rollouts from the base VLM
(default: Qwen3-VL-8B-Instruct), grade each rollout against the ground-truth
answer, and write back three fields:

- `predictions` — the `K` raw model outputs
- `correctness` — `K` booleans (`grader(pred, answer)`)
- `pass_rate`   — `mean(correctness)` ∈ [0, 1]

This `pass_rate` is the **difficulty signal** used by the
*difficulty-axis curriculum* in Section 4.5 of the paper.

## Layout

```
passK/
├── run_inference.py                  # main driver — K rollouts → grade → pass_rate
├── analyze_pass_rate.py              # post-hoc analysis
├── generate_curriculum_per_stage.py  # sort one capability's jsonl by pass_rate
├── generate_curriculum_merged.py     # interleave the three sorted lists
├── merge_results.py
├── run_stage{1,2,3}*.sh              # ready-to-run launch scripts
├── models/                           # vLLM-backed Qwen-VL client
├── datasets/                         # jsonl dataset loaders (vqa / textqa)
├── extractors/                       # \boxed{...} answer extractor
├── judges/                           # mathruler-based answer matcher
└── metrics/                          # pass_rate aggregation
```

## Quickstart

Run K=16 rollouts of Qwen3-VL-8B-Instruct on the perception jsonl, then attach pass-rate columns:

```bash
python run_inference.py \
    --model_name qwen3-vl \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --tensor_parallel_size 8 \
    --dataset_type vqa \
    --dataset_path <path>/perception_difficulty_curriculum.jsonl \
    --image_dir <path>/images \
    --k 16 \
    --temperature 0.7 \
    --max_new_tokens 1024 \
    --save_freq 100 \
    --output_path ./perception_with_passrate.jsonl
```

The `run_stage{1,2,3}*.sh` scripts in this directory are the exact commands we used in the paper.

## Producing a difficulty-ordered training file

After computing pass rates, sort within a capability bucket:

```bash
python generate_curriculum_per_stage.py \
    --input ./perception_with_passrate.jsonl \
    --output ./perception_difficulty_curriculum.jsonl
```

Or interleave the three sorted lists into a single capability × difficulty stream:

```bash
python generate_curriculum_merged.py \
    --perception ./perception_difficulty_curriculum.jsonl \
    --textual    ./textual_reasoning_difficulty_curriculum.jsonl \
    --visual     ./visual_reasoning_difficulty_curriculum.jsonl \
    --output     ./merged_difficulty_curriculum.jsonl
```

Plug the resulting jsonl into the EasyR1 launchers under
`training/examples/curriculum/` (those scripts pass `data.shuffle=false` so
the on-disk order is preserved).
