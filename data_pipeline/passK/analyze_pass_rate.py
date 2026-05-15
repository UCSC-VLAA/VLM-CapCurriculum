#!/usr/bin/env python3
"""
Analyze pass rate distribution from inference result files.

This script reads one or more JSONL result files and visualizes
the pass rate distribution as a histogram.

Usage:
    python analyze_pass_rate.py \
        --input_files results/perception_all.jsonl results/visual_reasoning_all.jsonl \
        --output_path figures/pass_rate_distribution.png
        
    # Single file
    python analyze_pass_rate.py \
        --input_files results/perception_all.jsonl \
        --output_path figures/perception_pass_rate.png
"""

import os
import json
import argparse
from typing import List, Dict, Any
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze pass rate distribution from inference result files"
    )
    
    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        required=True,
        help="One or more JSONL result files"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the figure (optional, will display if not provided)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=16,
        help="Number of inferences per sample (default: 16)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title for the plot"
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[12, 6],
        help="Figure size as width height (default: 12 6)"
    )
    
    return parser.parse_args()


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


def analyze_pass_rates(data: List[Dict[str, Any]], k: int) -> Dict[str, Any]:
    """
    Analyze pass rate distribution.
    
    Args:
        data: List of result dictionaries with 'pass_rate' field
        k: Number of inferences per sample
        
    Returns:
        Dictionary with analysis results
    """
    pass_rates = [item.get("pass_rate", 0) for item in data]
    
    # Convert pass_rate to number of correct answers (0 to k)
    # pass_rate = num_correct / k, so num_correct = pass_rate * k
    num_correct_list = [round(pr * k) for pr in pass_rates]
    
    # Count distribution
    distribution = Counter(num_correct_list)
    
    # Ensure all values from 0 to k are present
    for i in range(k + 1):
        if i not in distribution:
            distribution[i] = 0
    
    # Calculate statistics
    avg_pass_rate = np.mean(pass_rates) if pass_rates else 0
    median_pass_rate = np.median(pass_rates) if pass_rates else 0
    std_pass_rate = np.std(pass_rates) if pass_rates else 0
    
    # Calculate percentage of samples with at least one correct
    at_least_one_correct = sum(1 for nc in num_correct_list if nc > 0)
    all_correct = sum(1 for nc in num_correct_list if nc == k)
    all_incorrect = sum(1 for nc in num_correct_list if nc == 0)
    
    return {
        "pass_rates": pass_rates,
        "num_correct_list": num_correct_list,
        "distribution": distribution,
        "total_samples": len(data),
        "avg_pass_rate": avg_pass_rate,
        "median_pass_rate": median_pass_rate,
        "std_pass_rate": std_pass_rate,
        "at_least_one_correct": at_least_one_correct,
        "all_correct": all_correct,
        "all_incorrect": all_incorrect,
    }


def plot_distribution(
    analysis: Dict[str, Any],
    k: int,
    title: str = None,
    figsize: tuple = (12, 6),
    output_path: str = None,
    input_files: List[str] = None,
) -> None:
    """
    Plot pass rate distribution as a histogram.
    
    Args:
        analysis: Analysis results from analyze_pass_rates
        k: Number of inferences per sample
        title: Plot title
        figsize: Figure size
        output_path: Path to save the figure
        input_files: List of input file names for subtitle
    """
    distribution = analysis["distribution"]
    total_samples = analysis["total_samples"]
    
    # Prepare data for plotting
    x = list(range(k + 1))
    counts = [distribution[i] for i in x]
    percentages = [c / total_samples * 100 for c in counts]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar chart
    bars = ax.bar(x, counts, color='steelblue', edgecolor='black', alpha=0.8)
    
    # Add count and percentage labels on bars
    for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
        if count > 0:
            # Add count on top of bar
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                f'{count}',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )
            # Add percentage inside bar (if bar is tall enough)
            if pct > 3:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    f'{pct:.1f}%',
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='white',
                    fontweight='bold'
                )
    
    # Set labels and title
    ax.set_xlabel(f'Number of Correct Answers (out of {k})', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Pass Rate Distribution (K={k})', fontsize=14, fontweight='bold')
    
    # Set x-axis ticks
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Add statistics text box
    stats_text = (
        f"Total Samples: {total_samples}\n"
        f"Avg Pass Rate: {analysis['avg_pass_rate']:.4f} ({analysis['avg_pass_rate']*100:.2f}%)\n"
        f"Median Pass Rate: {analysis['median_pass_rate']:.4f}\n"
        f"Std Dev: {analysis['std_pass_rate']:.4f}\n"
        f"All Correct ({k}/{k}): {analysis['all_correct']} ({analysis['all_correct']/total_samples*100:.2f}%)\n"
        f"All Incorrect (0/{k}): {analysis['all_incorrect']} ({analysis['all_incorrect']/total_samples*100:.2f}%)\n"
        f"At Least 1 Correct: {analysis['at_least_one_correct']} ({analysis['at_least_one_correct']/total_samples*100:.2f}%)"
    )
    
    # Add text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(
        0.98, 0.97, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props,
        family='monospace'
    )
    
    # Add input files as subtitle if provided
    if input_files:
        file_names = [os.path.basename(f) for f in input_files]
        if len(file_names) <= 3:
            subtitle = f"Files: {', '.join(file_names)}"
        else:
            subtitle = f"Files: {', '.join(file_names[:3])} and {len(file_names)-3} more"
        ax.text(
            0.5, -0.12, subtitle,
            transform=ax.transAxes,
            fontsize=9,
            ha='center',
            style='italic',
            color='gray'
        )
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Pass Rate Distribution Analysis")
    print("=" * 60)
    print(f"Input files: {len(args.input_files)} file(s)")
    for f in args.input_files:
        print(f"  - {f}")
    print(f"K (inferences per sample): {args.k}")
    print("=" * 60)
    
    # Load all data
    print("\nLoading data...")
    all_data = []
    for filepath in args.input_files:
        if not os.path.isfile(filepath):
            print(f"Warning: File not found: {filepath}")
            continue
        data = load_jsonl(filepath)
        print(f"  Loaded {len(data)} samples from {os.path.basename(filepath)}")
        all_data.extend(data)
    
    if not all_data:
        print("Error: No data loaded!")
        return
    
    print(f"\nTotal samples: {len(all_data)}")
    
    # Analyze pass rates
    print("\nAnalyzing pass rates...")
    analysis = analyze_pass_rates(all_data, args.k)
    
    # Print distribution
    print("\nPass Rate Distribution:")
    print("-" * 40)
    print(f"{'Correct':>10} | {'Count':>10} | {'Percentage':>10}")
    print("-" * 40)
    for i in range(args.k + 1):
        count = analysis["distribution"][i]
        pct = count / analysis["total_samples"] * 100
        print(f"{i:>10} | {count:>10} | {pct:>9.2f}%")
    print("-" * 40)
    
    # Print statistics
    print("\nStatistics:")
    print(f"  Average Pass Rate: {analysis['avg_pass_rate']:.4f} ({analysis['avg_pass_rate']*100:.2f}%)")
    print(f"  Median Pass Rate: {analysis['median_pass_rate']:.4f}")
    print(f"  Standard Deviation: {analysis['std_pass_rate']:.4f}")
    print(f"  All Correct ({args.k}/{args.k}): {analysis['all_correct']} ({analysis['all_correct']/analysis['total_samples']*100:.2f}%)")
    print(f"  All Incorrect (0/{args.k}): {analysis['all_incorrect']} ({analysis['all_incorrect']/analysis['total_samples']*100:.2f}%)")
    print(f"  At Least 1 Correct: {analysis['at_least_one_correct']} ({analysis['at_least_one_correct']/analysis['total_samples']*100:.2f}%)")
    
    # Plot distribution
    print("\nGenerating plot...")
    plot_distribution(
        analysis,
        k=args.k,
        title=args.title,
        figsize=tuple(args.figsize),
        output_path=args.output_path,
        input_files=args.input_files,
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
