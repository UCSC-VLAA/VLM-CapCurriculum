#!/usr/bin/env bash
# One-shot reproduction of the paper's headline Qwen3-VL-8B-Staged result.
#
# Runs Stage 1 (perception) → Stage 2 (textual reasoning) → Stage 3 (visual
# reasoning) sequentially, automatically chaining checkpoints between stages.
#
# Pre-conditions
# --------------
# 1. EasyR1 is installed and EASYR1_HOME points at it.
# 2. The three Stage datasets are downloaded locally:
#       UCSC-VLAA/VLM-CapCurriculum-Perception-Data        → $VLMCC_STAGE1_TRAIN, $VLMCC_STAGE1_IMAGE_DIR
#       UCSC-VLAA/VLM-CapCurriculum-TextReasoning-Data     → $VLMCC_STAGE2_TRAIN
#       UCSC-VLAA/VLM-CapCurriculum-VisualReasoning-Data   → $VLMCC_STAGE3_TRAIN, $VLMCC_STAGE3_IMAGE_DIR
#    For each dataset, untar the *.tar.gz under images/ in place before
#    running. See each dataset's README on the Hub for the exact command.
# 3. 8 H100/H200 GPUs free (the script uses VLMCC_GPUS_PER_NODE).
#
# Usage
# -----
#   source training/_env.sh              # populate VLMCC_* paths
#   bash scripts/quickstart_qwen3vl_8b_staged.sh
#
# To resume after a stage, set START_STAGE:
#   START_STAGE=2 bash scripts/quickstart_qwen3vl_8b_staged.sh
#   START_STAGE=3 MODEL_PATH=<stage2-ckpt> bash scripts/quickstart_qwen3vl_8b_staged.sh
#
# Approximate wall-clock on 8× H200: 24 GPU-hours total.

set -euo pipefail

: "${EASYR1_HOME:?Run 'source training/_env.sh' first}"
: "${VLMCC_HOME:=$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")}"
: "${VLMCC_PROJECT:=VLM-CapCurriculum}"

START_STAGE=${START_STAGE:-1}
EXAMPLES="${VLMCC_HOME}/training/examples/qwen3_vl_8b"

# Resolve the latest checkpoint produced by an EasyR1 run, picking the highest
# global_step under <project>/<experiment>/.
latest_ckpt() {
    local exp_name=$1
    local base="${EASYR1_HOME}/checkpoints/${VLMCC_PROJECT}/${exp_name}"
    if [ ! -d "$base" ]; then
        echo ""; return
    fi
    ls -1d "${base}"/global_step_* 2>/dev/null \
        | sed 's|.*/global_step_||' | sort -n | tail -1 \
        | xargs -I{} echo "${base}/global_step_{}/actor/huggingface"
}

#--------------------------------------------------------------- Stage 1 ----
if [ "$START_STAGE" -le 1 ]; then
    echo
    echo "════════ Stage 1 / 3 — visual perception RLVR ════════"
    bash "${EXAMPLES}/stage1_perception.sh"

    STAGE1_CKPT=$(latest_ckpt "qwen3_vl_8b_stage1_perception")
    if [ -z "$STAGE1_CKPT" ] || [ ! -d "$STAGE1_CKPT" ]; then
        echo "  ERROR: could not locate Stage-1 checkpoint." >&2
        exit 1
    fi
    echo "  Stage 1 ✓ — checkpoint at: ${STAGE1_CKPT}"
else
    STAGE1_CKPT=${MODEL_PATH:-}
fi

#--------------------------------------------------------------- Stage 2 ----
if [ "$START_STAGE" -le 2 ]; then
    echo
    echo "════════ Stage 2 / 3 — textual reasoning RLVR ════════"
    : "${STAGE1_CKPT:?Stage-1 checkpoint not found}"
    MODEL_PATH="${STAGE1_CKPT}" bash "${EXAMPLES}/stage2_text_reasoning.sh"

    STAGE2_CKPT=$(latest_ckpt "qwen3_vl_8b_stage2_text_reasoning")
    if [ -z "$STAGE2_CKPT" ] || [ ! -d "$STAGE2_CKPT" ]; then
        echo "  ERROR: could not locate Stage-2 checkpoint." >&2
        exit 1
    fi
    echo "  Stage 2 ✓ — checkpoint at: ${STAGE2_CKPT}"
else
    STAGE2_CKPT=${MODEL_PATH:-}
fi

#--------------------------------------------------------------- Stage 3 ----
echo
echo "════════ Stage 3 / 3 — visual reasoning RLVR ════════"
: "${STAGE2_CKPT:?Stage-2 checkpoint not found}"
MODEL_PATH="${STAGE2_CKPT}" bash "${EXAMPLES}/stage3_visual_reasoning.sh"

STAGE3_CKPT=$(latest_ckpt "qwen3_vl_8b_stage3_visual_reasoning")
echo
echo "════════ Done — staged Qwen3-VL-8B trained ════════"
[ -n "$STAGE3_CKPT" ] && echo "  Final checkpoint: ${STAGE3_CKPT}"
echo
echo "Evaluate with:"
echo "  bash evaluation/serve_examples/serve_qwen3_vl_8b_staged.sh   # boot vLLM server"
echo "  bash evaluation/run_eval.sh Qwen3_VL_8B_Staged               # run benchmarks"
