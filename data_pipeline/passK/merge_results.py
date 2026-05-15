#!/usr/bin/env python3
"""
Merge checkpoint result files from multiple directories.

This script merges checkpoint JSONL files from one or more directories,
sorts them by index range, and outputs a single merged JSONL file.

Usage:
    python merge_results.py \
        --input_dirs results/stage1 results/stage2 results/stage3 \
        --output_path results/merged_output.jsonl
        
    # Or with a single directory
    python merge_results.py \
        --input_dirs results/checkpoints \
        --output_path results/final_output.jsonl
"""

import os
import re
import json
import argparse
from typing import List, Tuple, Dict, Any
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge checkpoint result files from multiple directories"
    )
    
    parser.add_argument(
        "--input_dirs",
        type=str,
        nargs="+",
        required=True,
        help="One or more directories containing checkpoint JSONL files"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the merged output JSONL file"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.jsonl",
        help="Glob pattern to match checkpoint files (default: *.jsonl)"
    )
    
    return parser.parse_args()


def extract_index_range(filename: str) -> Tuple[int, int]:
    """
    Extract start and end index from checkpoint filename.
    
    Expected format: xxx_START-END.jsonl
    e.g., visual_reasoning_train_11900-11999.jsonl -> (11900, 11999)
    
    Args:
        filename: Checkpoint filename
        
    Returns:
        Tuple of (start_idx, end_idx), or (-1, -1) if not found
    """
    # Match pattern like _123-456.jsonl
    match = re.search(r'_(\d+)-(\d+)\.jsonl$', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return -1, -1


def collect_checkpoint_files(input_dirs: List[str], pattern: str) -> List[Tuple[str, int, int]]:
    """
    Collect all checkpoint files from input directories.
    
    Args:
        input_dirs: List of directory paths
        pattern: Glob pattern to match files
        
    Returns:
        List of tuples (filepath, start_idx, end_idx)
    """
    checkpoint_files = []
    
    for input_dir in input_dirs:
        if not os.path.isdir(input_dir):
            print(f"Warning: Directory not found: {input_dir}")
            continue
        
        # Find all matching files
        file_pattern = os.path.join(input_dir, pattern)
        files = glob(file_pattern)
        
        for filepath in files:
            filename = os.path.basename(filepath)
            start_idx, end_idx = extract_index_range(filename)
            
            if start_idx >= 0:
                checkpoint_files.append((filepath, start_idx, end_idx))
            else:
                print(f"Warning: Could not parse index range from: {filename}")
    
    return checkpoint_files


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


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Merge Checkpoint Results")
    print("=" * 60)
    print(f"Input directories: {args.input_dirs}")
    print(f"Output path: {args.output_path}")
    print(f"File pattern: {args.pattern}")
    print("=" * 60)
    
    # Collect all checkpoint files
    print("\n[1/3] Collecting checkpoint files...")
    checkpoint_files = collect_checkpoint_files(args.input_dirs, args.pattern)
    
    if not checkpoint_files:
        print("Error: No checkpoint files found!")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    # Sort by start index
    checkpoint_files.sort(key=lambda x: x[1])
    
    # Print sorted checkpoint list
    print("\nCheckpoint files (sorted by index):")
    for filepath, start_idx, end_idx in checkpoint_files:
        print(f"  [{start_idx:6d} - {end_idx:6d}] {os.path.basename(filepath)}")
    
    # Check for gaps or overlaps
    print("\n[2/3] Checking for gaps and overlaps...")
    has_issues = False
    for i in range(1, len(checkpoint_files)):
        prev_file, prev_start, prev_end = checkpoint_files[i - 1]
        curr_file, curr_start, curr_end = checkpoint_files[i]
        
        if curr_start <= prev_end:
            print(f"  Warning: Overlap detected between {os.path.basename(prev_file)} and {os.path.basename(curr_file)}")
            has_issues = True
        elif curr_start > prev_end + 1:
            gap_start = prev_end + 1
            gap_end = curr_start - 1
            print(f"  Warning: Gap detected: indices {gap_start} - {gap_end} are missing")
            has_issues = True
    
    if not has_issues:
        print("  No gaps or overlaps detected")
    
    # Merge all files
    print("\n[3/3] Merging checkpoint files...")
    all_results = []
    
    for filepath, start_idx, end_idx in checkpoint_files:
        results = load_jsonl(filepath)
        all_results.extend(results)
        print(f"  Loaded {len(results)} samples from {os.path.basename(filepath)}")
    
    # Save merged results
    save_jsonl(all_results, args.output_path)
    
    # Calculate final index range
    if checkpoint_files:
        final_start_idx = checkpoint_files[0][1]
        final_end_idx = checkpoint_files[-1][2]
    else:
        final_start_idx = 0
        final_end_idx = 0
    
    # Print summary
    print("\n" + "=" * 60)
    print("Merge Summary")
    print("=" * 60)
    print(f"Total checkpoint files: {len(checkpoint_files)}")
    print(f"Total samples merged: {len(all_results)}")
    print(f"Index range: {final_start_idx} - {final_end_idx}")
    print(f"Output saved to: {args.output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
