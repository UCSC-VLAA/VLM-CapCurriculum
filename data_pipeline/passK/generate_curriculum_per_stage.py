#!/usr/bin/env python3
"""
Generate difficulty curriculum from inference result files.

This script reads one or more JSONL result files, sorts samples by pass_rate
(from high to low = easy to hard), and outputs curriculum files.

Usage:
    python generate_curriculum_per_stage.py \
        --input_files results/perception.jsonl results/textual_reasoning.jsonl results/visual_reasoning.jsonl \
        --output_dir curriculum/per_stage/
        
    # With custom suffix
    python generate_curriculum_per_stage.py \
        --input_files results/perception.jsonl \
        --output_dir curriculum/ \
        --suffix "_sorted"
"""

import os
import json
import random
import argparse
from typing import List, Dict, Any


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate difficulty curriculum from inference result files"
    )
    
    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        required=True,
        help="One or more JSONL result files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save curriculum files"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_curriculum",
        help="Suffix to add to output filenames (default: _curriculum)"
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling same pass_rate samples (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Handle sort order (default is descending = easy to hard)
    if args.ascending:
        args.descending = False
    
    return args


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL file.
    
    Args:
        filepath: Path to the JSONL file
        
    Returns:
        List of parsed JSON objects
    """
    results = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def save_jsonl(data: List[Dict[str, Any]], filepath: str) -> None:
    """
    Save data to a JSONL file.
    
    Args:
        data: List of dictionaries to save
        filepath: Path to save the file
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_output_filename(input_filepath: str, suffix: str, remove_unlearnable: bool = False) -> str:
    """
    Generate output filename based on input filename.
    
    Args:
        input_filepath: Input file path
        suffix: Suffix to add
        remove_unlearnable: Whether this is the filtered version
        
    Returns:
        Output filename (without directory)
    """
    basename = os.path.basename(input_filepath)
    name, ext = os.path.splitext(basename)
    
    if remove_unlearnable:
        return f"{name}{suffix}_filtered_new{ext}"
    else:
        return f"{name}{suffix}{ext}"


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


def process_file(
    input_filepath: str,
    output_dir: str,
    suffix: str,
    ascending: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Process a single file: sort by pass_rate and save both versions.
    
    Args:
        input_filepath: Path to input file
        output_dir: Directory to save output files
        suffix: Suffix for output filenames
        ascending: Sort order (True = hard to easy)
        seed: Random seed for shuffling same pass_rate samples
        
    Returns:
        Statistics dictionary
    """
    # Load data
    data = load_jsonl(input_filepath)
    total_samples = len(data)
    
    if total_samples == 0:
        return {
            "input_file": input_filepath,
            "total_samples": 0,
            "error": "Empty file"
        }
    
    # Sort by pass_rate with random shuffle for same pass_rate
    sorted_data = sort_by_pass_rate_random(data, ascending, seed)
    
    # Count pass_rate = 0 and pass_rate = 1
    pass_rate_zero = sum(1 for item in data if item.get("pass_rate", 0) == 0)
    pass_rate_one = sum(1 for item in data if item.get("pass_rate", 0) == 1)
    
    # Filter out pass_rate = 0 and pass_rate = 1
    # filtered_data = [
    #     item for item in sorted_data 
    #     if 0 < item.get("pass_rate", 0) < 1
    # ]
    # Remove samples with pass_rate = 1
    filtered_data = [
        item for item in sorted_data 
        if item.get("pass_rate", 0) != 1
    ]
    
    # Generate output paths
    output_filename = get_output_filename(input_filepath, suffix, remove_unlearnable=False)
    output_path = os.path.join(output_dir, output_filename)
    
    filtered_filename = get_output_filename(input_filepath, suffix, remove_unlearnable=True)
    filtered_path = os.path.join(output_dir, filtered_filename)
    
    # Save files
    save_jsonl(sorted_data, output_path)
    save_jsonl(filtered_data, filtered_path)
    
    return {
        "input_file": os.path.basename(input_filepath),
        "total_samples": total_samples,
        "pass_rate_zero": pass_rate_zero,
        "pass_rate_one": pass_rate_one,
        "filtered_samples": len(filtered_data),
        "output_file": output_filename,
        "filtered_file": filtered_filename,
    }


def main():
    args = parse_args()
    
    sort_order = "descending (easy to hard)" if args.descending else "ascending (hard to easy)"
    
    print("=" * 70)
    print("Generate Difficulty Curriculum")
    print("=" * 70)
    print(f"Input files: {len(args.input_files)} file(s)")
    for f in args.input_files:
        print(f"  - {f}")
    print(f"Output directory: {args.output_dir}")
    print(f"Suffix: {args.suffix}")
    print(f"Sort order: {sort_order}")
    print(f"Random seed: {args.seed}")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each file
    print("\nProcessing files...")
    all_stats = []
    
    for input_filepath in args.input_files:
        if not os.path.isfile(input_filepath):
            print(f"Warning: File not found: {input_filepath}")
            continue
        
        print(f"\nProcessing: {os.path.basename(input_filepath)}")
        stats = process_file(
            input_filepath=input_filepath,
            output_dir=args.output_dir,
            suffix=args.suffix,
            ascending=not args.descending,
            seed=args.seed,
        )
        all_stats.append(stats)
        
        if "error" in stats:
            print(f"  Error: {stats['error']}")
        else:
            print(f"  Total samples: {stats['total_samples']}")
            print(f"  Pass rate = 0 (unlearnable): {stats['pass_rate_zero']} ({stats['pass_rate_zero']/stats['total_samples']*100:.2f}%)")
            print(f"  Pass rate = 1 (too easy): {stats['pass_rate_one']} ({stats['pass_rate_one']/stats['total_samples']*100:.2f}%)")
            print(f"  Filtered samples: {stats['filtered_samples']} ({stats['filtered_samples']/stats['total_samples']*100:.2f}%)")
            print(f"  Output: {stats['output_file']}")
            print(f"  Filtered output: {stats['filtered_file']}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Input File':<40} | {'Total':>8} | {'PR=0':>8} | {'PR=1':>8} | {'Filtered':>8}")
    print("-" * 70)
    
    total_all = 0
    total_zero = 0
    total_one = 0
    total_filtered = 0
    
    for stats in all_stats:
        if "error" not in stats:
            print(f"{stats['input_file']:<40} | {stats['total_samples']:>8} | {stats['pass_rate_zero']:>8} | {stats['pass_rate_one']:>8} | {stats['filtered_samples']:>8}")
            total_all += stats['total_samples']
            total_zero += stats['pass_rate_zero']
            total_one += stats['pass_rate_one']
            total_filtered += stats['filtered_samples']
    
    print("-" * 70)
    print(f"{'TOTAL':<40} | {total_all:>8} | {total_zero:>8} | {total_one:>8} | {total_filtered:>8}")
    print("=" * 70)
    
    print(f"\nOutput files saved to: {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
