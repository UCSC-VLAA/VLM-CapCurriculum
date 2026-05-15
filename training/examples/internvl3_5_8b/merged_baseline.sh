#!/usr/bin/env bash
# Merged-training baseline for InternVL3.5-8B.
# Usage:  source training/_env.sh && bash training/examples/internvl3_5_8b/merged_baseline.sh
set -euo pipefail

: "${EASYR1_HOME:?Run 'source training/_env.sh' first}"
cd "${EASYR1_HOME}"

MODEL_PATH=${MODEL_PATH:-${VLMCC_BACKBONE_INTERNVL3_5_8B}}

python3 -m verl.trainer.main \
    config="${VLMCC_CONFIG}" \
    data.train_files="${VLMCC_MERGED_TRAIN}" \
    data.val_files=hiyouga/geometry3k@test \
    data.image_dir="${VLMCC_MERGED_IMAGE_DIR}" \
    data.format_prompt="${VLMCC_PROMPTS}/math.jinja" \
    data.max_prompt_length=2048 \
    data.max_pixels=500000 \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.model.freeze_vision_tower=false \
    worker.actor.offload.offload_params=false \
    worker.actor.offload.offload_optimizer=false \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=16 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.gpu_memory_utilization=0.7 \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_function="${VLMCC_REWARDS}/math.py:compute_score" \
    worker.reward.reward_type=batch \
    trainer.experiment_name=internvl3_5_8b_merged_baseline \
    trainer.project_name="${VLMCC_PROJECT}" \
    trainer.total_epochs=15 \
    trainer.val_freq=45 \
    trainer.save_freq=45 \
    trainer.save_limit=3 \
    trainer.n_gpus_per_node="${VLMCC_GPUS_PER_NODE}"
