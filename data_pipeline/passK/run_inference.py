#!/usr/bin/env python3
"""
Main inference script for pass rate evaluation.

This script runs VLM inference K times on each sample and computes pass rates.

Usage:
    python run_inference.py \
        --model_name qwen3-vl \
        --model_path Qwen/Qwen3-VL-8B-Instruct \
        --dataset_type vqa \
        --dataset_path /path/to/dataset.jsonl \
        --image_dir /path/to/images \
        --k 4 \
        --output_path results.jsonl \
        --save_freq 100 \
        --start_idx 0
"""

import os
import json
import argparse
import logging
from typing import Optional, List, Dict, Any
from tqdm import tqdm

# Suppress vLLM verbose logging
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
logging.getLogger("vllm").setLevel(logging.WARNING)

from models import get_model
from datasets import get_dataset
from extractors import get_extractor
from judges import get_judge
from metrics import compute_pass_rate


def parse_args():
    parser = argparse.ArgumentParser(
        description="VLM Inference Pipeline for Pass Rate Evaluation"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True,
        help="Model type name (e.g., qwen3-vl, qwen2.5-vl)"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to model (local dir or HuggingFace path)"
    )
    parser.add_argument(
        "--tensor_parallel_size", 
        type=int, 
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Maximum model context length"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        choices=["vqa", "text_qa", "auto"],
        help="Dataset type (vqa, text_qa, or auto)"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset file (JSONL format)"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Base directory for images (required for VQA)"
    )
    parser.add_argument(
        "--use_thinking",
        action="store_true",
        default=True,
        help="Use thinking-style system prompt"
    )
    parser.add_argument(
        "--no_thinking",
        action="store_true",
        help="Disable thinking-style system prompt"
    )
    
    # Inference arguments
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of inference runs per sample"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output JSONL file"
    )
    
    # Other arguments
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all)"
    )
    parser.add_argument(
        "--max_image_num",
        type=int,
        default=4,
        help="Maximum number of images per prompt"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start processing from this index (0-based)"
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=None,
        help="Save checkpoint every N samples (default: no checkpointing)"
    )
    
    args = parser.parse_args()
    
    # Handle thinking flag
    if args.no_thinking:
        args.use_thinking = False
    
    return args


def get_checkpoint_path(output_path: str, start_idx: int, end_idx: int) -> str:
    """
    Generate checkpoint file path with index range in filename.
    
    Args:
        output_path: Original output path
        start_idx: Start index (inclusive)
        end_idx: End index (inclusive)
        
    Returns:
        Checkpoint file path with index range
    """
    base, ext = os.path.splitext(output_path)
    return f"{base}_{start_idx}-{end_idx}{ext}"


def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save results to JSONL file.
    
    Args:
        results: List of result dictionaries
        output_path: Path to save the file
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("PassK Inference Pipeline")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"K: {args.k}")
    print(f"Output: {args.output_path}")
    print(f"Start Index: {args.start_idx}")
    if args.save_freq:
        print(f"Save Frequency: every {args.save_freq} samples")
    print("=" * 60)
    
    # Initialize model
    print("\n[1/4] Loading model...")
    model = get_model(
        model_name=args.model_name,
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_image_num=args.max_image_num,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )
    print(f"Model loaded: {args.model_name}")
    
    # Load dataset
    print("\n[2/4] Loading dataset...")
    dataset_kwargs = {
        "dataset_name": args.dataset_type,
        "data_path": args.dataset_path,
        "use_thinking": args.use_thinking,
    }
    if args.image_dir:
        dataset_kwargs["image_dir"] = args.image_dir
    
    dataset = get_dataset(**dataset_kwargs)
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Get samples with start_idx and num_samples
    all_samples = list(dataset)
    
    # Apply start_idx
    if args.start_idx > 0:
        if args.start_idx >= len(all_samples):
            print(f"Error: start_idx ({args.start_idx}) >= dataset size ({len(all_samples)})")
            return
        all_samples = all_samples[args.start_idx:]
        print(f"Starting from index {args.start_idx}")
    
    # Apply num_samples limit
    if args.num_samples is not None:
        all_samples = all_samples[:args.num_samples]
    
    samples = all_samples
    total_samples = len(samples)
    print(f"Processing {total_samples} samples (indices {args.start_idx} to {args.start_idx + total_samples - 1})")
    
    # Initialize extractor and judge
    print("\n[3/4] Initializing extractor and judge...")
    extractor = get_extractor("boxed")
    judge = get_judge("mathruler")
    print("Extractor: boxed (mathruler)")
    print("Judge: mathruler")
    
    # Get system prompt
    system_prompt = dataset.get_system_prompt()
    print(f"\nSystem prompt: {system_prompt[:100]}...")
    
    # Run inference
    print(f"\n[4/4] Running inference (K={args.k})...")
    results = []
    checkpoint_results = []  # Results since last checkpoint
    last_checkpoint_idx = args.start_idx
    
    # Overall progress bar (position 0)
    overall_pbar = tqdm(
        total=total_samples,
        desc="Overall Progress",
        position=0,
        leave=True,
    )
    
    # Per-sample K inference progress bar (position 1)
    k_pbar = tqdm(
        total=args.k,
        desc="K Inference",
        position=1,
        leave=False,
    )
    
    for sample_idx, sample in enumerate(samples):
        # Global index in the original dataset
        global_idx = args.start_idx + sample_idx
        
        # Update overall progress bar description
        overall_pbar.set_description(f"Sample {sample_idx + 1}/{total_samples} (idx={global_idx})")
        
        # Reset K progress bar
        k_pbar.reset()
        k_pbar.set_description(f"K Inference [{sample.index}]")
        
        # Get prompt and images
        prompt = dataset.get_prompt(sample)
        images = dataset.get_images(sample)
        
        # Generate K responses (batch generation)
        responses = model.generate_k_times(
            prompt=prompt,
            k=args.k,
            images=images,
            system_prompt=system_prompt,
        )
        
        # Extract answers and judge correctness
        predictions = []
        correctness = []
        
        for i, response in enumerate(responses):
            # Extract answer
            pred_answer = extractor.extract(response)
            predictions.append(pred_answer)
            
            # Judge correctness
            is_correct = judge.judge(pred_answer, sample.answer)
            correctness.append(is_correct)
            
            # Update K progress bar
            k_pbar.update(1)
            k_pbar.set_postfix({"correct": sum(correctness), "pred": pred_answer[:20] + "..." if len(pred_answer) > 20 else pred_answer})
        
        # Compute pass rate
        pass_rate = compute_pass_rate(correctness)
        
        # Build result (only keep extracted answers, correctness, and pass_rate)
        result = sample.to_dict()
        
        # Convert images back to relative paths (remove image_dir prefix)
        if args.image_dir and result.get("images"):
            relative_images = []
            for img_path in result["images"]:
                if img_path.startswith(args.image_dir):
                    # Remove image_dir prefix and leading slash
                    rel_path = img_path[len(args.image_dir):].lstrip(os.sep)
                    relative_images.append(rel_path)
                else:
                    relative_images.append(img_path)
            result["images"] = relative_images
        
        result["predictions"] = predictions
        result["correctness"] = correctness
        result["pass_rate"] = pass_rate
        
        results.append(result)
        checkpoint_results.append(result)
        
        # Update overall progress bar
        overall_pbar.update(1)
        avg_pass_rate = sum(r["pass_rate"] for r in results) / len(results)
        overall_pbar.set_postfix({"avg_pass_rate": f"{avg_pass_rate:.2%}"})
        
        # Save checkpoint if needed
        if args.save_freq and len(checkpoint_results) >= args.save_freq:
            checkpoint_end_idx = global_idx
            checkpoint_path = get_checkpoint_path(args.output_path, last_checkpoint_idx, checkpoint_end_idx)
            save_results(checkpoint_results, checkpoint_path)
            print(f"\n[Checkpoint] Saved {len(checkpoint_results)} samples to {checkpoint_path}")
            
            # Reset for next checkpoint
            checkpoint_results = []
            last_checkpoint_idx = global_idx + 1
    
    # Close progress bars
    k_pbar.close()
    overall_pbar.close()
    
    # Save remaining results as checkpoint if there are any
    if args.save_freq and checkpoint_results:
        checkpoint_end_idx = args.start_idx + total_samples - 1
        checkpoint_path = get_checkpoint_path(args.output_path, last_checkpoint_idx, checkpoint_end_idx)
        save_results(checkpoint_results, checkpoint_path)
        print(f"\n[Checkpoint] Saved {len(checkpoint_results)} samples to {checkpoint_path}")
    
    # Save all results to final output
    print(f"\nSaving all results to {args.output_path}...")
    save_results(results, args.output_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    num_results = len(results)
    avg_pass_rate = sum(r["pass_rate"] for r in results) / num_results if num_results > 0 else 0
    print(f"Total samples: {num_results}")
    print(f"Index range: {args.start_idx} - {args.start_idx + num_results - 1}")
    print(f"Average pass rate: {avg_pass_rate:.4f} ({avg_pass_rate*100:.2f}%)")
    print(f"Results saved to: {args.output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
