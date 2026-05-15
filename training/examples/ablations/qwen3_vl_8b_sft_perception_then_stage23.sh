#!/usr/bin/env bash
# SFT vs RLVR for Stage 1 (perception) — caption-SFT baseline (Table 6).
#
# Stage 1 SFT is run with LLaMA-Factory (not EasyR1) because GRPO is replaced
# by next-token captioning supervision. After SFT, Stage 2 and Stage 3 follow
# the regular RLVR scripts. This shell helper documents the wiring; the SFT
# part needs to be launched manually with the LLaMA-Factory installation.
#
#   1. Build the caption-SFT dataset for DOCCI (one image / "Please generate
#      a detailed caption" / caption-as-target). See data_pipeline/README.md.
#   2. Run LLaMA-Factory full-parameter SFT on Qwen3-VL-8B-Instruct, e.g.:
#         llamafactory-cli train \\
#             --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \\
#             --dataset docci_caption_dataset \\
#             --finetuning_type full --stage sft \\
#             --output_dir <SFT_CHECKPOINT>
#   3. Use <SFT_CHECKPOINT> as MODEL_PATH for Stage 2:
#         MODEL_PATH=<SFT_CHECKPOINT> bash training/examples/qwen3_vl_8b/stage2_text_reasoning.sh
#   4. Then Stage 3 as usual:
#         MODEL_PATH=<STAGE2_CKPT> bash training/examples/qwen3_vl_8b/stage3_visual_reasoning.sh
#
# This script is intentionally print-only — running it just prints the recipe.
cat <<'EOF'
SFT vs RLVR ablation (paper Section 4.4 / Table 6) — recipe:

 1) Build caption-SFT dataset from DOCCI captions:
       python data_pipeline/sft/make_caption_sft.py \
           --captions /path/to/docci_descriptions.jsonl \
           --image-root /path/to/DOCCI/images_downsampled_2x \
           --output /path/to/docci_caption_dataset.json
    (or reuse generate_vlm_caption_dataset.py from the original codebase)

 2) Stage-1 caption-SFT with LLaMA-Factory:
       llamafactory-cli train \
           --model_name_or_path  Qwen/Qwen3-VL-8B-Instruct \
           --dataset             docci_caption_dataset \
           --finetuning_type     full \
           --stage               sft \
           --output_dir          <SFT_CHECKPOINT>

 3) Stage 2 (text reasoning) initialised from the SFT checkpoint:
       MODEL_PATH=<SFT_CHECKPOINT> \
           bash training/examples/qwen3_vl_8b/stage2_text_reasoning.sh

 4) Stage 3 (visual reasoning) initialised from Stage-2 checkpoint:
       MODEL_PATH=<STAGE2_CHECKPOINT> \
           bash training/examples/qwen3_vl_8b/stage3_visual_reasoning.sh
EOF
