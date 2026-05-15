#!/usr/bin/env bash
# Stage 3: visual reasoning RLVR for Qwen3-VL-8B.
# Initialise from the Stage-2 checkpoint via MODEL_PATH.
# Usage:  source training/_env.sh && bash training/examples/qwen3_vl_8b/stage3_visual_reasoning.sh
set -euo pipefail

: "${EASYR1_HOME:?Run 'source training/_env.sh' first}"
cd "${EASYR1_HOME}"

# After Stage 2 finishes, point MODEL_PATH at:
#   <EASYR1_HOME>/checkpoints/<project>/qwen3_vl_8b_stage2_text_reasoning/global_step_NN/actor/huggingface
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to the Stage-2 checkpoint}

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
    trainer.experiment_name=qwen3_vl_8b_stage3_visual_reasoning \
    trainer.project_name="${VLMCC_PROJECT}" \
    trainer.total_epochs=15 \
    trainer.val_freq=15 \
    trainer.save_freq=31 \
    trainer.save_limit=3 \
    trainer.n_gpus_per_node="${VLMCC_GPUS_PER_NODE}"
