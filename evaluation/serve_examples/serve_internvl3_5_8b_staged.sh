#!/usr/bin/env bash
# Serve the released InternVL3.5-8B-Staged checkpoint via LMDeploy.
set -euo pipefail

MODEL=${MODEL:-UCSC-VLAA/VLM-CapCurriculum-InternVL3.5-8B-Staged}
PORT=${PORT:-23343}
TP=${TP:-4}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} \
    lmdeploy serve api_server "${MODEL}" \
        --server-port "${PORT}" \
        --tp "${TP}"
