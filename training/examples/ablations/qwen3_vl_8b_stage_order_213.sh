#!/usr/bin/env bash
# Stage-order ablation 2→1→3 on Qwen3-VL-8B (Table 5).
# Pre-requisite: a Stage-2 → Stage-1 chained checkpoint, then run Stage 3 below.
# Set MODEL_PATH to the post-Stage-1 checkpoint of that chain.
set -euo pipefail

: "${EASYR1_HOME:?Run 'source training/_env.sh' first}"
cd "${EASYR1_HOME}"

MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to the Stage 2→1 checkpoint}

python3 -m verl.trainer.main \
    config="${VLMCC_CONFIG}" \
    data.train_files="${VLMCC_STAGE3_TRAIN}" \
    data.val_files="${VLMCC_STAGE3_VAL}" \
    data.image_dir="${VLMCC_STAGE3_IMAGE_DIR}" \
    data.format_prompt="${VLMCC_PROMPTS}/math.jinja" \
    data.max_prompt_length=2048 \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.model.freeze_vision_tower=false \
    worker.actor.offload.offload_params=false \
    worker.actor.offload.offload_optimizer=false \
    worker.actor.micro_batch_size_per_device_for_update=16 \
    worker.actor.micro_batch_size_per_device_for_experience=64 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.gpu_memory_utilization=0.7 \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_function="${VLMCC_REWARDS}/math.py:compute_score" \
    worker.reward.reward_type=batch \
    trainer.experiment_name=qwen3_vl_8b_stage_order_2_1_3 \
    trainer.project_name="${VLMCC_PROJECT}" \
    trainer.total_epochs=15 \
    trainer.val_freq=15 \
    trainer.save_freq=31 \
    trainer.save_limit=3 \
    trainer.n_gpus_per_node="${VLMCC_GPUS_PER_NODE}"
