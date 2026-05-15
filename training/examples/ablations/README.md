# Ablation runs

Each script here mirrors a row in the ablation tables of the paper.
Set `MODEL_PATH` (when needed) to the appropriate checkpoint and source
`training/_env.sh` first.

| File | Maps to |
|---|---|
| `qwen3_vl_8b_stage3_only.sh` | Table 2 row "Stage 3" — apply only Stage 3 to the base model |
| `qwen3_vl_8b_stage1_stage3.sh` | Table 2 row "Stage 1→3" |
| `qwen3_vl_8b_stage2_stage3.sh` | Table 2 row "Stage 2→3" |
| `qwen3_vl_8b_stage_order_213.sh` | Table 5 row "2→1→3" (stage-order) |
| `qwen3_vl_8b_stage_order_321.sh` | Table 5 row "3→2→1" (reversed stage order) |
| `qwen3_vl_8b_merged_freeze_vision.sh` | Appendix Table — merged with frozen vision encoder |
| `qwen2_5_vl_7b_stage3_only.sh` | Table 2 / Table 5 base+stage3 row |
| `qwen2_5_vl_7b_stage_order_213.sh` | Table 5 row "2→1→3" on Qwen2.5 |
| `qwen2_5_vl_7b_stage_order_321.sh` | Table 5 row "3→2→1" on Qwen2.5 |
| `qwen2_5_vl_7b_merged_freeze_vision.sh` | Appendix Table — frozen vision baseline |
| `qwen3_vl_8b_sft_perception_then_stage23.sh` | Table 6 — caption-SFT replaces Stage 1 RLVR |

For the stage-order variants the script naming reads `1→2→3` left-to-right.
The merged-with-perception baseline (Section 3.4 column "Merged + perception")
uses the same data as `merged_baseline.sh`, only the dataset itself is built
with perception included; we don't ship a separate launch script for it.
