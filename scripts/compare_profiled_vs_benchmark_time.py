#!/usr/bin/env python3

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def parse_timing_comparison(output_text: str) -> Dict[str, float]:
    """
    Compare total profiled time vs benchmark total time.
    """
    data = {}
    
    # Extract total benchmark time in nanoseconds
    benchmark_match = re.search(r'BM_ycsb.*?\s([\d.e+]+)\s+ns\s+([\d.e+]+)\s+ns\s+(\d+)\s+', output_text)
    if benchmark_match:
        real_time_ns = float(benchmark_match.group(1))
        cpu_time_ns = float(benchmark_match.group(2))
        iterations = int(benchmark_match.group(3))
        total_benchmark_time_ns = real_time_ns * iterations
        data['benchmark_time_ns'] = total_benchmark_time_ns
        data['real_time_ns'] = real_time_ns
        data['iterations'] = iterations
    
    # Extract thread count from benchmark name
    thread_match = re.search(r'threads:(\d+)', output_text)
    if thread_match:
        data['thread_count'] = int(thread_match.group(1))
    else:
        data['thread_count'] = 1
    
    # Parse BATCH MIGRATION section
    batch_section = re.search(r'--- BATCH MIGRATION ---.*?(?=--- SINGLE-PAGE MIGRATION ---)', output_text, re.DOTALL)
    if batch_section:
        batch_text = batch_section.group(0)
        
        # Extract samples and total timing
        samples_match = re.search(r'Samples:\s*([\d]+)', batch_text)
        total_timing_match = re.search(r'TOTAL:\s+([\d.]+)\s+us', batch_text)
        
        if samples_match and total_timing_match:
            batch_samples = int(samples_match.group(1))
            batch_total_us = float(total_timing_match.group(1))
            batch_total_ns = batch_total_us * 1000 * batch_samples
            
            data['batch_samples'] = batch_samples
            data['batch_avg_timing_us'] = batch_total_us
            data['batch_total_ns'] = batch_total_ns
    
    # Parse SINGLE-PAGE MIGRATION section
    single_section = re.search(r'--- SINGLE-PAGE MIGRATION ---.*?(?====)', output_text, re.DOTALL)
    if single_section:
        single_text = single_section.group(0)
        
        # Extract samples and total timing
        samples_match = re.search(r'Samples:\s*([\d]+)', single_text)
        total_timing_match = re.search(r'TOTAL:\s+([\d.]+)\s+us', single_text)
        
        if samples_match and total_timing_match:
            single_samples = int(samples_match.group(1))
            single_total_us = float(total_timing_match.group(1))
            single_total_ns = single_total_us * 1000 * single_samples
            
            data['single_samples'] = single_samples
            data['single_avg_timing_us'] = single_total_us
            data['single_total_ns'] = single_total_ns
    
    # Calculate totals
    if 'batch_total_ns' in data and 'single_total_ns' in data:
        data['profiled_total_ns'] = data['batch_total_ns'] + data['single_total_ns']
        
        if 'benchmark_time_ns' in data:
            data['profiled_vs_benchmark_ratio'] = data['profiled_total_ns'] / data['benchmark_time_ns']
    
    return data


def load_timing_comparison_data(timestamps: List[str], base_dir: Path) -> List[Tuple[str, Dict]]:
    """
    Load timing comparison data for given timestamps.
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
                
                comparison = parse_timing_comparison(output_text)
                
                if comparison:
                    data.append((run_id, comparison))
    
    return data


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare profiled migration time vs total benchmark time."
    )
    parser.add_argument(
        "--timestamps",
        nargs="+",
        default=["20260203_052559"],
        help="List of timestamps to analyze.",
    )
    parser.add_argument(
        "--base-dir",
        default="/users/yrayhan/hyrise/benchmark_results",
        help="Directory containing benchmark results.",
    )
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    print(f"Loading timing comparison data for timestamps: {args.timestamps}\n")
    data = load_timing_comparison_data(args.timestamps, base_dir)
    
    if not data:
        print("No timing data found. Exiting.")
        return
    
    print(f"{'Run ID':<50} {'Benchmark Time':<20} {'Profiled Time':<20} {'Profiled/Threads':<20} {'Match %':<10}")
    print("=" * 120)
    
    for run_id, comparison in data:
        benchmark_time = comparison.get('benchmark_time_ns', 0)
        profiled_time = comparison.get('profiled_total_ns', 0)
        thread_count = comparison.get('thread_count', 1)
        
        benchmark_ms = benchmark_time / 1e6
        profiled_ms = profiled_time / 1e6
        profiled_per_thread_ms = profiled_ms / thread_count  # Divide by actual thread count
        
        match_pct = (profiled_per_thread_ms / benchmark_ms * 100) if benchmark_ms > 0 else 0
        
        print(f"{run_id:<50} {benchmark_ms:>12.2f} ms    {profiled_ms:>12.2f} ms    {profiled_per_thread_ms:>15.2f} ms    {match_pct:>8.1f}%")
        
        # Detailed breakdown
        batch_samples = comparison.get('batch_samples', 0)
        single_samples = comparison.get('single_samples', 0)
        batch_avg_us = comparison.get('batch_avg_timing_us', 0)
        single_avg_us = comparison.get('single_avg_timing_us', 0)
        
        print(f"  Batch:  {batch_samples:>10,} samples × {batch_avg_us:>8.2f} us = {comparison.get('batch_total_ns', 0)/1e6:>10.2f} ms")
        print(f"  Single: {single_samples:>10,} samples × {single_avg_us:>8.2f} us = {comparison.get('single_total_ns', 0)/1e6:>10.2f} ms")
        print()


if __name__ == "__main__":
    main()
