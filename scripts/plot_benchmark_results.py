#!/usr/bin/env python3

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def parse_benchmark_metrics(output_text: str) -> Dict[str, float]:
    """
    Parse benchmark output to extract metrics.
    Expected format:
    BM_ycsb/... Time CPU Iterations bytes_per_second=X bytes_read_from_ssd=X ...
    """
    metrics = {}
    
    # Look for the benchmark result line
    pattern = r'BM_ycsb.*?latency_stddev=([\d.]+[kMG]?)'
    match = re.search(pattern, output_text)
    
    if not match:
        return metrics
    
    # Extract all key=value pairs
    metric_pattern = r'(\w+)=([\d.]+[kMG]?)'
    for key, value in re.findall(metric_pattern, output_text):
        # Convert value to float, handling k/M/G suffixes
        try:
            if value.endswith('k'):
                metrics[key] = float(value[:-1]) * 1000
            elif value.endswith('M'):
                metrics[key] = float(value[:-1]) * 1_000_000
            elif value.endswith('G'):
                metrics[key] = float(value[:-1]) * 1_000_000_000
            else:
                metrics[key] = float(value)
        except ValueError:
            pass
    
    # Extract total syscalls (both batch and single)
    total_syscalls = 0
    
    # Look for batch syscalls in BATCH MIGRATION section
    batch_syscalls_match = re.search(r'--- BATCH MIGRATION ---.*?Total syscalls:\s*([\d]+)', output_text, re.DOTALL)
    if batch_syscalls_match:
        total_syscalls += int(batch_syscalls_match.group(1))
    
    # Look for single syscalls in SINGLE-PAGE MIGRATION section
    single_syscalls_match = re.search(r'--- SINGLE-PAGE MIGRATION ---.*?Total syscalls:\s*([\d]+)', output_text, re.DOTALL)
    if single_syscalls_match:
        total_syscalls += int(single_syscalls_match.group(1))
    
    if total_syscalls > 0:
        metrics['total_syscalls'] = float(total_syscalls)
    
    return metrics


def load_benchmark_data(timestamps: List[str], base_dir: Path) -> Dict[str, List[Tuple[str, Dict]]]:
    """
    Load benchmark results for given timestamps.
    Returns: {metric_name: [(run_id, metrics_dict), ...]}
    """
    all_metrics = {}
    
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
                
                metrics = parse_benchmark_metrics(output_text)
                
                if not metrics:
                    print(f"  Warning: No metrics found in {output_file}")
                    continue
                
                # Store metrics by type
                for metric_name, value in metrics.items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append((run_id, value))
    
    return all_metrics


def create_bar_plots(all_metrics: Dict[str, List[Tuple[str, float]]], output_dir: Path):
    """
    Create separate bar plots for each metric.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Metrics to plot (filtering out some that might be redundant)
    metrics_to_plot = [
        "bytes_per_second",
        "bytes_read_from_ssd",
        "bytes_written_to_ssd",
        "cache_hit_rate",
        "items_per_second",
        "latency_95percentile",
        "latency_max",
        "latency_mean",
        "latency_median",
        "latency_min",
        "latency_stddev",
        "total_syscalls",
    ]
    
    for metric_name in metrics_to_plot:
        if metric_name not in all_metrics:
            print(f"Metric {metric_name} not found in data")
            continue
        
        data = all_metrics[metric_name]
        run_ids = [item[0] for item in data]
        values = [item[1] for item in data]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Create bar plot
        x_pos = np.arange(len(run_ids))
        bars = ax.bar(x_pos, values, color='mediumseagreen', edgecolor='darkgreen', alpha=0.7)
        
        # Customize plot
        ax.set_xlabel("Run Configuration", fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f"Benchmark Metric: {metric_name}", fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(run_ids, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars (vertical)
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.2e}' if height >= 1e6 else f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=8,
                rotation=90
            )
        
        plt.tight_layout()
        
        # Save figure
        safe_metric_name = metric_name.replace('/', '_')
        output_path = output_dir / f"benchmark_{safe_metric_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        
        plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create bar plots from benchmark results."
    )
    parser.add_argument(
        "--timestamps",
        nargs="+",
        default=["20260203_052559", "20260202_205624"],
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
    
    print(f"Loading benchmark data for timestamps: {args.timestamps}")
    all_metrics = load_benchmark_data(args.timestamps, base_dir)
    
    if not all_metrics:
        print("No metrics found. Exiting.")
        return
    
    print(f"\nFound metrics: {list(all_metrics.keys())}")
    print(f"\nCreating plots...")
    create_bar_plots(all_metrics, output_dir)
    
    print(f"\nDone! Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
