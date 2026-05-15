#!/usr/bin/env bash
# Stage-order ablation 3→2→1 on Qwen3-VL-8B (Table 5).
# This is the "reversed" run: visual reasoning → textual reasoning → perception.
# It is exposed as one shell helper that simply executes Stage 1 (perception) on
# top of a pre-trained Stage-3 → Stage-2 chain. To launch the full reversed
# trajectory, sequentially run:
#   bash ablations/qwen3_vl_8b_stage3_only.sh         (initial 3)
#   bash qwen3_vl_8b/stage2_text_reasoning.sh         (then 2 on Stage-3 ckpt)
#   bash ablations/qwen3_vl_8b_stage_order_321.sh     (this script: 1 on top)
set -euo pipefail

: "${EASYR1_HOME:?Run 'source training/_env.sh' first}"
cd "${EASYR1_HOME}"

MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to the Stage 3→2 checkpoint}

python3 -m verl.trainer.main \
    config="${VLMCC_CONFIG}" \
    data.train_files="${VLMCC_STAGE1_TRAIN}" \
    data.val_files=hiyouga/geometry3k@test \
    data.image_dir="${VLMCC_STAGE1_IMAGE_DIR}" \
    data.format_prompt="${VLMCC_PROMPTS}/math.jinja" \
    data.prompt_key=problem \
    data.image_key=images \
    data.max_prompt_length=2048 \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.model.freeze_vision_tower=false \
    worker.actor.offload.offload_params=false \
    worker.actor.offload.offload_optimizer=false \
    worker.actor.micro_batch_size_per_device_for_update=16 \
    worker.actor.micro_batch_size_per_device_for_experience=32 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.gpu_memory_utilization=0.7 \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_function="${VLMCC_REWARDS}/math.py:compute_score" \
    worker.reward.reward_type=batch \
    trainer.experiment_name=qwen3_vl_8b_stage_order_3_2_1 \
    trainer.project_name="${VLMCC_PROJECT}" \
    trainer.total_epochs=16 \
    trainer.val_freq=6 \
    trainer.save_freq=12 \
    trainer.save_limit=3 \
    trainer.n_gpus_per_node="${VLMCC_GPUS_PER_NODE}"
