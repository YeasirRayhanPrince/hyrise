#!/usr/bin/env python3

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def parse_total_syscalls(output_text: str) -> float:
    """
    Extract total system calls (both batch and single combined) from benchmark output.
    """
    total_syscalls = 0
    
    # Look for batch syscalls in BATCH MIGRATION section
    batch_syscalls_match = re.search(r'--- BATCH MIGRATION ---.*?Total syscalls:\s*([\d]+)', output_text, re.DOTALL)
    if batch_syscalls_match:
        total_syscalls += int(batch_syscalls_match.group(1))
    
    # Look for single syscalls in SINGLE-PAGE MIGRATION section
    single_syscalls_match = re.search(r'--- SINGLE-PAGE MIGRATION ---.*?Total syscalls:\s*([\d]+)', output_text, re.DOTALL)
    if single_syscalls_match:
        total_syscalls += int(single_syscalls_match.group(1))
    
    return float(total_syscalls)


def load_syscalls_data(timestamps: List[str], base_dir: Path) -> List[Tuple[str, float]]:
    """
    Load total syscalls for given timestamps.
    Returns: [(run_id, syscalls_count), ...]
    """
    data = []
    
    for timestamp in timestamps:
        summary_path = base_dir / f"summary_{timestamp}.jsonl"
        
        if not summary_path.exists():
            print(f"Warning: {summary_path} not found")
            continue
        
        print(f"Loading {summary_path}...")
        
        with summary_path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                run_id = record.get("run_id", "unknown")
                output_file = record.get("output_file")
                
                if not output_file:
                    continue
                
                output_path = Path(output_file)
                if not output_path.exists():
                    print(f"  Warning: Output file {output_file} not found")
                    continue
                
                # Read and parse benchmark output
                with output_path.open("r", encoding="utf-8") as out_f:
                    output_text = out_f.read()
                
                syscalls = parse_total_syscalls(output_text)
                
                if syscalls > 0:
                    data.append((run_id, syscalls))
    
    return data


def create_syscalls_plot(data: List[Tuple[str, float]], output_dir: Path):
    """
    Create a bar plot for total system calls, scaled relative to the first bar.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_ids = [item[0] for item in data]
    values = [item[1] for item in data]
    
    # Normalize values with respect to the first bar
    first_value = values[0]
    normalized_values = [v / first_value for v in values]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create bar plot
    x_pos = np.arange(len(run_ids))
    bars = ax.bar(x_pos, normalized_values, color='mediumseagreen', edgecolor='darkgreen', alpha=0.7)
    
    # Customize plot
    ax.set_xlabel("Run Configuration", fontsize=12)
    ax.set_ylabel("Relative Total System Calls (scaled to first bar)", fontsize=12)
    ax.set_title("Total System Calls (Batch + Single) - Normalized", fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(run_ids, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (first bar)')
    
    # Add value labels on bars (vertical)
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.2f}x',
            ha='center',
            va='bottom',
            fontsize=8,
            rotation=90
        )
    
    ax.legend()
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "benchmark_total_syscalls.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create a plot for total system calls from benchmark results."
    )
    parser.add_argument(
        "--timestamps",
        nargs="+",
        default=["20260203_052559"],
        help="List of timestamps to plot.",
    )
    parser.add_argument(
        "--base-dir",
        default="/users/yrayhan/hyrise/benchmark_results",
        help="Directory containing benchmark results.",
    )
    parser.add_argument(
        "--output-dir",
        default="/users/yrayhan/hyrise/benchmark_plots",
        help="Directory to save plots.",
    )
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    
    print(f"Loading syscalls data for timestamps: {args.timestamps}")
    data = load_syscalls_data(args.timestamps, base_dir)
    
    if not data:
        print("No syscalls data found. Exiting.")
        return
    
    print(f"Found {len(data)} runs with syscalls data")
    print(f"Creating plot...")
    create_syscalls_plot(data, output_dir)
    
    print(f"Done! Plot saved to {output_dir}")


if __name__ == "__main__":
    main()
