#!/usr/bin/env bash
# Serve the released Qwen3-VL-8B-Staged checkpoint via vLLM.
# The port (23341) MUST match the one in evaluation/configs/models.py.
set -euo pipefail

MODEL=${MODEL:-UCSC-VLAA/VLM-CapCurriculum-Qwen3-VL-8B-Staged}      # or local path
PORT=${PORT:-23341}
TP=${TP:-4}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} \
    vllm serve "${MODEL}" \
        --tensor-parallel-size "${TP}" \
        --gpu-memory-utilization 0.9 \
        --port "${PORT}"
