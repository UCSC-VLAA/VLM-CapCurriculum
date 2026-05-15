#!/usr/bin/env bash
# Stage 2: textual reasoning RLVR for InternVL3-8B.
#
# IMPORTANT: InternVL3 needs damped optimization to avoid entropy explosion —
# smaller LR (3e-7), larger KL coefficient (5e-2), tighter clipping (0.15),
# and reduced grad-norm bound (0.5). Do NOT remove these knobs.
#
# Usage:  source training/_env.sh && bash training/examples/internvl3_8b/stage2_text_reasoning.sh
set -euo pipefail

: "${EASYR1_HOME:?Run 'source training/_env.sh' first}"
cd "${EASYR1_HOME}"

MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to the Stage-1 checkpoint}

python3 -m verl.trainer.main \
    config="${VLMCC_CONFIG}" \
    data.train_files="${VLMCC_STAGE2_TRAIN}" \
    data.val_files=hiyouga/geometry3k@test \
    data.format_prompt="${VLMCC_PROMPTS}/math.jinja" \
    data.max_prompt_length=2048 \
    algorithm.kl_coef=5.0e-2 \
    worker.actor.optim.lr=3.0e-7 \
    worker.actor.optim.lr_warmup_ratio=0.05 \
    worker.actor.global_batch_size=128 \
    worker.actor.max_grad_norm=0.5 \
    worker.actor.clip_ratio_high=0.15 \
    worker.actor.clip_ratio_low=0.15 \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.model.freeze_vision_tower=true \
    worker.actor.offload.offload_params=false \
    worker.actor.offload.offload_optimizer=false \
    worker.actor.micro_batch_size_per_device_for_update=8 \
    worker.actor.micro_batch_size_per_device_for_experience=32 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.gpu_memory_utilization=0.7 \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_function="${VLMCC_REWARDS}/math.py:compute_score" \
    worker.reward.reward_type=batch \
    trainer.experiment_name=internvl3_8b_stage2_text_reasoning \
    trainer.project_name="${VLMCC_PROJECT}" \
    trainer.total_epochs=15 \
    trainer.val_freq=25 \
    trainer.save_freq=25 \
    trainer.save_limit=3 \
    trainer.n_gpus_per_node="${VLMCC_GPUS_PER_NODE}"
