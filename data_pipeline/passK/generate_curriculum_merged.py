#!/usr/bin/env python3
"""
Generate merged difficulty curriculum from multiple inference result files.

This script reads multiple JSONL result files, merges all samples,
sorts by pass_rate, and outputs merged curriculum files.

Two sorting modes for samples with same pass_rate:
1. Random shuffle
2. By input file order (file1 samples first, then file2, etc.)

Usage:
    python generate_curriculum_merged.py \
        --input_files results/perception.jsonl results/textual_reasoning.jsonl results/visual_reasoning.jsonl \
        --output_dir curriculum/merged/
"""

import os
import json
import random
import argparse
from typing import List, Dict, Any, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate merged difficulty curriculum from multiple files"
    )
    
    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        required=True,
        help="One or more JSONL result files (order matters for file-order sorting)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save merged curriculum files"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="merged",
        help="Prefix for output filenames (default: merged)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)"
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort by pass_rate ascending (hard to easy)"
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        default=True,
        help="Sort by pass_rate descending (easy to hard, default for curriculum learning)"
    )
    
    args = parser.parse_args()
    
    # Handle sort order (default is descending = easy to hard)
    if args.ascending:
        args.descending = False
    
    return args


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load a JSONL file."""
    results = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def save_jsonl(data: List[Dict[str, Any]], filepath: str) -> None:
    """Save data to a JSONL file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_all_data(input_files: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Load data from all input files and add file order info.
    
    Args:
        input_files: List of input file paths
        
    Returns:
        Tuple of (all_data, file_stats)
    """
    all_data = []
    file_stats = {}
    
    for file_idx, filepath in enumerate(input_files):
        if not os.path.isfile(filepath):
            print(f"Warning: File not found: {filepath}")
            continue
        
        data = load_jsonl(filepath)
        filename = os.path.basename(filepath)
        file_stats[filename] = len(data)
        
        # Add file order and original index to each sample
        for sample_idx, sample in enumerate(data):
            sample["_file_order"] = file_idx
            sample["_sample_order"] = sample_idx
            sample["_source_file"] = filename
            all_data.append(sample)
        
        print(f"  Loaded {len(data)} samples from {filename}")
    
    return all_data, file_stats


def sort_by_pass_rate_random(
    data: List[Dict[str, Any]], 
    ascending: bool = True,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Sort by pass_rate, with random shuffle for same pass_rate.
    
    Args:
        data: List of samples
        ascending: Sort order
        seed: Random seed
        
    Returns:
        Sorted list
    """
    random.seed(seed)
    
    # Group by pass_rate
    groups = {}
    for sample in data:
        pr = sample.get("pass_rate", 0)
        if pr not in groups:
            groups[pr] = []
        groups[pr].append(sample)
    
    # Shuffle within each group
    for pr in groups:
        random.shuffle(groups[pr])
    
    # Sort pass_rates
    sorted_prs = sorted(groups.keys(), reverse=not ascending)
    
    # Flatten
    result = []
    for pr in sorted_prs:
        result.extend(groups[pr])
    
    return result


def sort_by_pass_rate_file_order(
    data: List[Dict[str, Any]], 
    ascending: bool = True,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Sort by pass_rate, with file order for same pass_rate.
    Within each (pass_rate, file_order) group, samples are randomly shuffled.
    
    Args:
        data: List of samples with _file_order and _sample_order
        ascending: Sort order
        seed: Random seed for shuffling within same (pass_rate, file_order) group
        
    Returns:
        Sorted list
    """
    random.seed(seed)
    
    # Group by (pass_rate, file_order)
    groups = {}
    for sample in data:
        pr = sample.get("pass_rate", 0)
        file_order = sample.get("_file_order", 0)
        key = (pr, file_order)
        if key not in groups:
            groups[key] = []
        groups[key].append(sample)
    
    # Shuffle within each group
    for key in groups:
        random.shuffle(groups[key])
    
    # Sort keys by (pass_rate, file_order)
    sorted_keys = sorted(groups.keys(), key=lambda x: (x[0], x[1]), reverse=not ascending)
    
    # Flatten
    result = []
    for key in sorted_keys:
        result.extend(groups[key])
    
    return result


# def filter_unlearnable(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """Filter out samples with pass_rate = 0 or pass_rate = 1."""
#     return [
#         sample for sample in data
#         if 0 < sample.get("pass_rate", 0) < 1
#     ]

def filter_unlearnable(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove samples with pass_rate = 1."""
    return [
        sample for sample in data
        if sample.get("pass_rate", 0) != 1
    ]



def clean_internal_fields(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove internal fields (_file_order, _sample_order, _source_file)."""
    cleaned = []
    for sample in data:
        sample_copy = {k: v for k, v in sample.items() if not k.startswith("_")}
        cleaned.append(sample_copy)
    return cleaned


def main():
    args = parse_args()
    
    sort_order = "descending (easy to hard)" if args.descending else "ascending (hard to easy)"
    
    print("=" * 70)
    print("Generate Merged Difficulty Curriculum")
    print("=" * 70)
    print(f"Input files: {len(args.input_files)} file(s)")
    for i, f in enumerate(args.input_files):
        print(f"  [{i}] {f}")
    print(f"Output directory: {args.output_dir}")
    print(f"Output prefix: {args.output_prefix}")
    print(f"Sort order: {sort_order}")
    print(f"Random seed: {args.seed}")
    print("=" * 70)
    
    # Load all data
    print("\nLoading data...")
    all_data, file_stats = load_all_data(args.input_files)
    
    if not all_data:
        print("Error: No data loaded!")
        return
    
    total_samples = len(all_data)
    print(f"\nTotal samples: {total_samples}")
    
    # Count pass_rate = 0 and pass_rate = 1
    pass_rate_zero = sum(1 for s in all_data if s.get("pass_rate", 0) == 0)
    pass_rate_one = sum(1 for s in all_data if s.get("pass_rate", 0) == 1)
    filtered_count = total_samples - pass_rate_zero - pass_rate_one
    
    print(f"Pass rate = 0 (unlearnable): {pass_rate_zero} ({pass_rate_zero/total_samples*100:.2f}%)")
    print(f"Pass rate = 1 (too easy): {pass_rate_one} ({pass_rate_one/total_samples*100:.2f}%)")
    print(f"Learnable (0 < PR < 1): {filtered_count} ({filtered_count/total_samples*100:.2f}%)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # === Version 1: Random shuffle for same pass_rate ===
    print("\n" + "-" * 70)
    print("Version 1: Random shuffle for same pass_rate")
    print("-" * 70)
    
    sorted_random = sort_by_pass_rate_random(all_data.copy(), not args.descending, args.seed)
    sorted_random_filtered = filter_unlearnable(sorted_random)
    
    # Clean internal fields before saving
    sorted_random_clean = clean_internal_fields(sorted_random)
    sorted_random_filtered_clean = clean_internal_fields(sorted_random_filtered)
    
    # Save
    random_path = os.path.join(args.output_dir, f"{args.output_prefix}_random.jsonl")
    random_filtered_path = os.path.join(args.output_dir, f"{args.output_prefix}_random_filtered_new.jsonl")
    
    save_jsonl(sorted_random_clean, random_path)
    save_jsonl(sorted_random_filtered_clean, random_filtered_path)
    
    print(f"  Full: {len(sorted_random_clean)} samples -> {random_path}")
    print(f"  Filtered: {len(sorted_random_filtered_clean)} samples -> {random_filtered_path}")
    
    # === Version 2: File order for same pass_rate ===
    print("\n" + "-" * 70)
    print("Version 2: File order for same pass_rate")
    print("-" * 70)
    
    sorted_file_order = sort_by_pass_rate_file_order(all_data.copy(), not args.descending, args.seed)
    sorted_file_order_filtered = filter_unlearnable(sorted_file_order)
    
    # Clean internal fields before saving
    sorted_file_order_clean = clean_internal_fields(sorted_file_order)
    sorted_file_order_filtered_clean = clean_internal_fields(sorted_file_order_filtered)
    
    # Save
    file_order_path = os.path.join(args.output_dir, f"{args.output_prefix}_file_order.jsonl")
    file_order_filtered_path = os.path.join(args.output_dir, f"{args.output_prefix}_file_order_filtered.jsonl")
    
    save_jsonl(sorted_file_order_clean, file_order_path)
    save_jsonl(sorted_file_order_filtered_clean, file_order_filtered_path)
    
    print(f"  Full: {len(sorted_file_order_clean)} samples -> {file_order_path}")
    print(f"  Filtered: {len(sorted_file_order_filtered_clean)} samples -> {file_order_filtered_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total samples: {total_samples}")
    print(f"Filtered samples: {filtered_count}")
    print("\nOutput files:")
    print(f"  1. {args.output_prefix}_random.jsonl ({len(sorted_random_clean)} samples)")
    print(f"  2. {args.output_prefix}_random_filtered.jsonl ({len(sorted_random_filtered_clean)} samples)")
    print(f"  3. {args.output_prefix}_internal_stage_order.jsonl ({len(sorted_file_order_clean)} samples)")
    print(f"  4. {args.output_prefix}_internal_stage_order_filtered.jsonl ({len(sorted_file_order_filtered_clean)} samples)")
    print("=" * 70)
    print("\nDone!")


if __name__ == "__main__":
    main()
