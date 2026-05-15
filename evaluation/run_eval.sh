#!/usr/bin/env bash
# Run the paper's evaluation suite on one (or more) checkpoints via VLMEvalKit.
#
#   bash evaluation/run_eval.sh <model-alias> [<model-alias> ...]
#
# Where <model-alias> is one of the keys registered in evaluation/configs/models.py
# (e.g. Qwen3_VL_8B_Staged). The aliases must already be merged into VLMEvalKit's
# vlmeval/config.py (see evaluation/README.md → "How to install").
#
# This script assumes:
#   - VLMEvalKit is on PYTHONPATH (or installed)   → set $VLMEVALKIT_HOME
#   - The Bedrock judge patch is applied           → see vlmevalkit_patches/
#   - LMDeploy servers backing each alias are up   → see configs/models.py
#   - AWS credentials for Bedrock are configured   → aws configure / env vars

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <model-alias> [<model-alias> ...]" >&2
    echo "  e.g. $0 Qwen3_VL_8B_Staged Qwen3_VL_8B_Merged" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Required env
# ---------------------------------------------------------------------------
: "${VLMEVALKIT_HOME:?Set VLMEVALKIT_HOME to your VLMEvalKit clone}"
: "${VLMCC_HOME:=$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")}"

cd "${VLMEVALKIT_HOME}"

# ---------------------------------------------------------------------------
# Benchmark groups (matches the paper)
#   Main     — Table 1 main comparison
#   Extended — Appendix Table 9 (extended benchmarks)
#   Curriculum — Section 4.5, Table 7
#   All      — union (deduplicated)
# ---------------------------------------------------------------------------
MAIN_BENCH=(
    MathVista_MINI
    MathVision_MINI
    MathVerse_MINI_Vision_Intensive
    WeMath
    A-OKVQA
    RealWorldQA
    MMStar
    POPE
)

EXTENDED_BENCH=(
    MathVerse_MINI_Vision_Only
    DynaMath
    HallusionBench
    BLINK
    VisOnlyQA-Synthetic
    VisOnlyQA-Real
    CV-Bench-2D
    VStarBench
    ChartQA_TEST
    TextVQA_VAL
)

# Default: run the main suite. Switch with VLMCC_BENCH=extended|all|<custom>.
case "${VLMCC_BENCH:-main}" in
    main)        DATASETS=("${MAIN_BENCH[@]}") ;;
    extended)    DATASETS=("${EXTENDED_BENCH[@]}") ;;
    all)         DATASETS=("${MAIN_BENCH[@]}" "${EXTENDED_BENCH[@]}") ;;
    *)           # space-separated custom override
                 IFS=' ' read -r -a DATASETS <<< "${VLMCC_BENCH}" ;;
esac

JUDGE=${VLMCC_JUDGE:-bedrock-claude-haiku-4.5}
NPROC=${VLMCC_NPROC:-8}
EXTRA_ARGS=${VLMCC_EXTRA_ARGS:-}

echo "VLMEvalKit         : ${VLMEVALKIT_HOME}"
echo "Benchmarks         : ${DATASETS[*]}"
echo "Judge              : ${JUDGE}"
echo "API parallelism    : ${NPROC}"
echo "Models             : $*"
echo

for MODEL in "$@"; do
    echo "──── ${MODEL} ────"
    python run.py \
        --data "${DATASETS[@]}" \
        --model "${MODEL}" \
        --judge "${JUDGE}" \
        --api-nproc "${NPROC}" \
        --verbose \
        ${EXTRA_ARGS}
done

echo "Done. Per-model result files are under ${VLMEVALKIT_HOME}/outputs/<MODEL>/"
