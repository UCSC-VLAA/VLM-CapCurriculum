#!/usr/bin/env bash
# Serve the released InternVL3-8B-Staged checkpoint via LMDeploy
# (InternVL family is served via LMDeploy in our setup).
set -euo pipefail

MODEL=${MODEL:-UCSC-VLAA/VLM-CapCurriculum-InternVL3-8B-Staged}
PORT=${PORT:-23342}
TP=${TP:-4}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} \
    lmdeploy serve api_server "${MODEL}" \
        --server-port "${PORT}" \
        --tp "${TP}"
