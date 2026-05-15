# Curriculum experiments (paper Section 4.5, Table 7)

Four configurations on Qwen3-VL-8B that probe the *capability axis × difficulty
axis* design space. The only practical difference between them is the input
data file (already pre-ordered) and whether `data.shuffle` is `true` or
`false`. All four use the merged-baseline hyper-parameters.

| Curriculum | Data ordering | Script |
|---|---|---|
| **None** (Merged) | random shuffle | `qwen3_vl_8b/merged_baseline.sh` — the standard merged script |
| **Capability only** | three sequential stages (== our staged 1→2→3 recipe) | `qwen3_vl_8b/stage{1,2,3}_*.sh` *(or `qwen3_vl_8b_capability_only.sh` if you prefer one launcher consuming a pre-stitched file)* |
| **Difficulty only** | merged corpus sorted by sample difficulty | `qwen3_vl_8b_difficulty_only.sh` |
| **Capability + Difficulty** | difficulty-sorted within each capability bucket | `qwen3_vl_8b_capability_x_difficulty.sh` |

All curriculum-table scripts pass `data.shuffle=false` so the on-disk order is
preserved during training. Set `VLMCC_DIFFICULTY_*` env vars (see `_env.sh`)
to point at the pre-ordered jsonl files.
