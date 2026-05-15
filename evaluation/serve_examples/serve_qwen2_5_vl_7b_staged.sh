#!/usr/bin/env bash
# Serve the released Qwen2.5-VL-7B-Staged checkpoint via vLLM.
set -euo pipefail

MODEL=${MODEL:-UCSC-VLAA/VLM-CapCurriculum-Qwen2.5-VL-7B-Staged}
PORT=${PORT:-23340}
TP=${TP:-4}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} \
    vllm serve "${MODEL}" \
        --tensor-parallel-size "${TP}" \
        --gpu-memory-utilization 0.9 \
        --port "${PORT}"
