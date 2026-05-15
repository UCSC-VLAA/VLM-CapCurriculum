#!/usr/bin/env bash
# Stage 1: visual perception RLVR for InternVL3-8B.
# Note: InternVL3 uses a smaller per-device batch and larger max_prompt_length
# than the Qwen scripts to fit memory.
# Usage:  source training/_env.sh && bash training/examples/internvl3_8b/stage1_perception.sh
set -euo pipefail

: "${EASYR1_HOME:?Run 'source training/_env.sh' first}"
cd "${EASYR1_HOME}"

MODEL_PATH=${MODEL_PATH:-${VLMCC_BACKBONE_INTERNVL3_8B}}

python3 -m verl.trainer.main \
    config="${VLMCC_CONFIG}" \
    data.train_files="${VLMCC_STAGE1_TRAIN}" \
    data.val_files=hiyouga/geometry3k@test \
    data.image_dir="${VLMCC_STAGE1_IMAGE_DIR}" \
    data.format_prompt="${VLMCC_PROMPTS}/math.jinja" \
    data.prompt_key=problem \
    data.image_key=images \
    data.max_prompt_length=4096 \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.model.freeze_vision_tower=false \
    worker.actor.offload.offload_params=false \
    worker.actor.offload.offload_optimizer=true \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.gpu_memory_utilization=0.7 \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_function="${VLMCC_REWARDS}/math.py:compute_score" \
    worker.reward.reward_type=batch \
    trainer.experiment_name=internvl3_8b_stage1_perception \
    trainer.project_name="${VLMCC_PROJECT}" \
    trainer.total_epochs=16 \
    trainer.val_freq=6 \
    trainer.save_freq=12 \
    trainer.save_limit=3 \
    trainer.n_gpus_per_node="${VLMCC_GPUS_PER_NODE}"
