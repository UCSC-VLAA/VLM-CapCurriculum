#!/usr/bin/env bash
# End-to-end Stage 1 perception data construction:
#   captions  →  generate MCQs  →  filter (image-only wrong AND caption-only right
#                                          for two VLMs)  →  format for EasyR1.
#
# Edit the paths at the top, then run from the data_pipeline/ root.
#
#   bash examples/run_full_pipeline.sh

set -euo pipefail

# ---- knobs ---------------------------------------------------------------
SOURCE=${SOURCE:-docci}                   # docci | pixmo
CAPTIONS=${CAPTIONS:-/path/to/docci_descriptions.jsonl}
IMAGE_ROOT=${IMAGE_ROOT:-/path/to/DOCCI/images_downsampled_2x}
OUT_DIR=${OUT_DIR:-./outputs}

GEN_MODEL=${GEN_MODEL:-Qwen/Qwen2.5-72B-Instruct}
FILTER_MODEL_A=${FILTER_MODEL_A:-Qwen/Qwen2.5-VL-7B-Instruct}
FILTER_MODEL_B=${FILTER_MODEL_B:-Qwen/Qwen2.5-VL-32B-Instruct}

TP_GEN=${TP_GEN:-8}
TP_FILTER=${TP_FILTER:-8}

# ---- paths ---------------------------------------------------------------
mkdir -p "${OUT_DIR}"
MCQ_RAW=${OUT_DIR}/${SOURCE}_mcq_raw.jsonl
PRED_IMG_A=${OUT_DIR}/${SOURCE}_pred_img_A.jsonl
PRED_IMG_B=${OUT_DIR}/${SOURCE}_pred_img_B.jsonl
PRED_CAP_A=${OUT_DIR}/${SOURCE}_pred_cap_A.jsonl
PRED_CAP_B=${OUT_DIR}/${SOURCE}_pred_cap_B.jsonl
MCQ_FILTERED=${OUT_DIR}/${SOURCE}_perception_filtered.jsonl
TRAIN_DIR=${OUT_DIR}/${SOURCE}_training_jsonl

# ---- 1. generation -------------------------------------------------------
python generate_qa.py \
    --captions "${CAPTIONS}" \
    --source   "${SOURCE}" \
    --model    "${GEN_MODEL}" \
    --output   "${MCQ_RAW}" \
    --tensor-parallel-size "${TP_GEN}"

# ---- 2. image-only inference (two models) --------------------------------
python filter_perception.py --mode image \
    --mcq "${MCQ_RAW}" \
    --model "${FILTER_MODEL_A}" --image-root "${IMAGE_ROOT}" \
    --output "${PRED_IMG_A}" --tensor-parallel-size "${TP_FILTER}"

python filter_perception.py --mode image \
    --mcq "${MCQ_RAW}" \
    --model "${FILTER_MODEL_B}" --image-root "${IMAGE_ROOT}" \
    --output "${PRED_IMG_B}" --tensor-parallel-size "${TP_FILTER}"

# ---- 3. caption-only inference (two models) ------------------------------
python filter_perception.py --mode caption \
    --mcq "${MCQ_RAW}" \
    --model "${FILTER_MODEL_A}" \
    --output "${PRED_CAP_A}" --tensor-parallel-size "${TP_FILTER}"

python filter_perception.py --mode caption \
    --mcq "${MCQ_RAW}" \
    --model "${FILTER_MODEL_B}" \
    --output "${PRED_CAP_B}" --tensor-parallel-size "${TP_FILTER}"

# ---- 4. set intersection -------------------------------------------------
python filter_perception.py --mode filter \
    --mcq "${MCQ_RAW}" \
    --image-preds   "${PRED_IMG_A}" "${PRED_IMG_B}" \
    --caption-preds "${PRED_CAP_A}" "${PRED_CAP_B}" \
    --output "${MCQ_FILTERED}"

# ---- 5. format for EasyR1 ------------------------------------------------
python format_for_training.py \
    --input "${MCQ_FILTERED}" \
    --output-dir "${TRAIN_DIR}" \
    --prefix "stage1_perception"

echo
echo "Done. Plug into EasyR1:"
echo "  data.train_files=${TRAIN_DIR}/train_stage1_perception.jsonl"
echo "  data.val_files=${TRAIN_DIR}/val_stage1_perception.jsonl"
echo "  data.image_dir=${IMAGE_ROOT}"
