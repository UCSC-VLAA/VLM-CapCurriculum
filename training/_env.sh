# Source this file before launching any training script.
#
#   source training/_env.sh
#
# Override anything below by exporting the variable yourself first.
# All training/examples/*.sh resolve their paths through these env vars,
# which is how we keep one set of scripts working across machines and
# data layouts.

# ---------------------------------------------------------------------------
# Repository roots
# ---------------------------------------------------------------------------
# VLMCC_HOME  = root of this repo (the directory containing this file's parent)
# EASYR1_HOME = where you cloned https://github.com/hiyouga/EasyR1
export VLMCC_HOME=${VLMCC_HOME:-$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")}
export EASYR1_HOME=${EASYR1_HOME:-$(realpath "${VLMCC_HOME}/../EasyR1")}

# ---------------------------------------------------------------------------
# Paper-side assets — point at the files copied from this repo
# ---------------------------------------------------------------------------
export VLMCC_REWARDS=${VLMCC_REWARDS:-${VLMCC_HOME}/training/reward_functions}
export VLMCC_PROMPTS=${VLMCC_PROMPTS:-${VLMCC_HOME}/training/format_prompts}
export VLMCC_CONFIG=${VLMCC_CONFIG:-${VLMCC_HOME}/training/configs/config.yaml}

# ---------------------------------------------------------------------------
# Data — set these to where your training jsonl + image roots live.
# Defaults assume the layout produced by data_pipeline/ + the public
# stage2/stage3 mixes used in the paper.
# ---------------------------------------------------------------------------
# Stage 1: perception MCQs (UCSC-VLAA/VLM-CapCurriculum-Perception)
export VLMCC_STAGE1_TRAIN=${VLMCC_STAGE1_TRAIN:-/path/to/Perception/perception_difficulty_curriculum.jsonl}
export VLMCC_STAGE1_IMAGE_DIR=${VLMCC_STAGE1_IMAGE_DIR:-/path/to/Perception/images}

# Stage 2: textual reasoning (UCSC-VLAA/VLM-CapCurriculum-TextReasoning, ORZ-Math-13k)
export VLMCC_STAGE2_TRAIN=${VLMCC_STAGE2_TRAIN:-/path/to/TextReasoning/textual_reasoning_difficulty_curriculum.jsonl}

# Stage 3: visual reasoning (UCSC-VLAA/VLM-CapCurriculum-VisualReasoning)
export VLMCC_STAGE3_TRAIN=${VLMCC_STAGE3_TRAIN:-/path/to/VisualReasoning/visual_reasoning_difficulty_curriculum.jsonl}
export VLMCC_STAGE3_VAL=${VLMCC_STAGE3_VAL:-hiyouga/geometry3k@test}
export VLMCC_STAGE3_IMAGE_DIR=${VLMCC_STAGE3_IMAGE_DIR:-/path/to/VisualReasoning/images}

# Merged baseline & capability×difficulty curriculum: produced locally by
# concatenating the three Stage jsonl files (and optionally sorting by
# pass_rate). See training/examples/curriculum/README.md for the recipe.
export VLMCC_MERGED_TRAIN=${VLMCC_MERGED_TRAIN:-/path/to/merged_train_shuffled.jsonl}
export VLMCC_MERGED_IMAGE_DIR=${VLMCC_MERGED_IMAGE_DIR:-/path/to/merged_images_root}

# Difficulty-curriculum versions (sorted by pass_rate within each file).
export VLMCC_DIFFICULTY_STAGE1_TRAIN=${VLMCC_DIFFICULTY_STAGE1_TRAIN:-${VLMCC_STAGE1_TRAIN}}
export VLMCC_DIFFICULTY_STAGE2_TRAIN=${VLMCC_DIFFICULTY_STAGE2_TRAIN:-${VLMCC_STAGE2_TRAIN}}
export VLMCC_DIFFICULTY_STAGE3_TRAIN=${VLMCC_DIFFICULTY_STAGE3_TRAIN:-${VLMCC_STAGE3_TRAIN}}
export VLMCC_DIFFICULTY_MERGED_TRAIN=${VLMCC_DIFFICULTY_MERGED_TRAIN:-/path/to/merged_difficulty.jsonl}
export VLMCC_DIFFICULTY_CAP_DIFF_TRAIN=${VLMCC_DIFFICULTY_CAP_DIFF_TRAIN:-/path/to/cap_x_diff.jsonl}

# ---------------------------------------------------------------------------
# Backbone HF ids — override locally to point at on-disk weights
# ---------------------------------------------------------------------------
export VLMCC_BACKBONE_QWEN2_5_VL_7B=${VLMCC_BACKBONE_QWEN2_5_VL_7B:-Qwen/Qwen2.5-VL-7B-Instruct}
export VLMCC_BACKBONE_QWEN3_VL_8B=${VLMCC_BACKBONE_QWEN3_VL_8B:-Qwen/Qwen3-VL-8B-Instruct}
export VLMCC_BACKBONE_INTERNVL3_8B=${VLMCC_BACKBONE_INTERNVL3_8B:-OpenGVLab/InternVL3-8B-hf}
export VLMCC_BACKBONE_INTERNVL3_5_8B=${VLMCC_BACKBONE_INTERNVL3_5_8B:-OpenGVLab/InternVL3_5-8B-HF}

# ---------------------------------------------------------------------------
# Run-time
# ---------------------------------------------------------------------------
export VLMCC_PROJECT=${VLMCC_PROJECT:-VLM-CapCurriculum}
export VLMCC_GPUS_PER_NODE=${VLMCC_GPUS_PER_NODE:-8}

export PYTHONUNBUFFERED=1
