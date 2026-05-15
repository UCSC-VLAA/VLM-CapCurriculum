#!/bin/bash
# Test script for VQA pass rate evaluation
# Uses Qwen3-VL-8B-Instruct model on perception_200.jsonl dataset
# Uses 8 GPUs for tensor parallel inference

cd /fsx/juncheng/passK

python run_inference.py \
    --model_name qwen3-vl \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --tensor_parallel_size 8 \
    --dataset_type vqa \
    --dataset_path /fsx-shared/juncheng/dataset/VLM/merge_stage/perception_200.jsonl \
    --image_dir /fsx-shared/juncheng/dataset/VLM/ \
    --k 4 \
    --temperature 0.7 \
    --max_new_tokens 1024 \
    --num_samples 5 \
    --output_path results/vqa_test_results.jsonl
