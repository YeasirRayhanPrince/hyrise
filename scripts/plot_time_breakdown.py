#!/usr/bin/env python3

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


# Define consistent color mapping for all components
def get_component_colors():
    """
    Return a consistent color mapping for all components across all plots.
    """
    components = [
        'batch_queue_scan', 'batch_locking', 'batch_grouping', 'batch_dirty_check',
        'batch_array_build', 'batch_syscall', 'batch_metadata_update', 'batch_unlock',
        'single_queue_scan', 'single_locking', 'single_grouping', 'single_dirty_check',
        'single_array_build', 'single_syscall', 'single_metadata_update', 'single_unlock',
        'other_unaccounted',
    ]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(components)))
    return {comp: colors[i] for i, comp in enumerate(components)}


COMPONENT_COLORS = get_component_colors()


def parse_time_breakdown(output_text: str) -> Dict[str, float]:
    """
    Parse timing breakdown from migration profiler output.
    Returns: {component_name: time_in_ns, ...}
    """
    timings = {}
    
    # Extract total benchmark time in nanoseconds
    benchmark_match = re.search(r'BM_ycsb.*?\s([\d.e+]+)\s+ns\s+([\d.e+]+)\s+ns\s+(\d+)\s+', output_text)
    if not benchmark_match:
        return timings
    
    real_time_ns = float(benchmark_match.group(1))
    iterations = int(benchmark_match.group(3))
    total_benchmark_time_ns = real_time_ns * iterations
    
    # Extract thread count from benchmark name
    thread_match = re.search(r'threads:(\d+)', output_text)
    thread_count = int(thread_match.group(1)) if thread_match else 1
    
    # Use total benchmark time (wall-clock) directly - don't divide by thread count
    timings['_total_benchmark_time_ns'] = total_benchmark_time_ns
    timings['_thread_count'] = thread_count
    
    # Parse BATCH MIGRATION section
    batch_section = re.search(r'--- BATCH MIGRATION ---.*?(?=--- SINGLE-PAGE MIGRATION ---)', output_text, re.DOTALL)
    if batch_section:
        batch_text = batch_section.group(0)
        
        # Extract samples
        samples_match = re.search(r'Samples:\s*([\d]+)', batch_text)
        if samples_match:
            batch_samples = int(samples_match.group(1))
            
            # Extract individual timings
            timing_patterns = {
                'batch_queue_scan': r'Queue scan:\s+([\d.]+)',
                'batch_locking': r'Locking:\s+([\d.]+)',
                'batch_grouping': r'Grouping:\s+([\d.]+)',
                'batch_dirty_check': r'Dirty check:\s+([\d.]+)',
                'batch_array_build': r'Array build:\s+([\d.]+)',
                'batch_syscall': r'Syscall:\s+([\d.]+)',
                'batch_metadata_update': r'Metadata update:\s+([\d.]+)',
                'batch_unlock': r'Unlock:\s+([\d.]+)',
            }
            
            for key, pattern in timing_patterns.items():
                match = re.search(pattern, batch_text)
                if match:
                    us_time = float(match.group(1))
                    ns_time = us_time * 1000 * batch_samples / thread_count  # Convert us to ns, multiply by samples, divide by thread count
                    timings[key] = ns_time
    
    # Parse SINGLE-PAGE MIGRATION section
    single_section = re.search(r'--- SINGLE-PAGE MIGRATION ---.*?(?====)', output_text, re.DOTALL)
    if single_section:
        single_text = single_section.group(0)
        
        # Extract samples
        samples_match = re.search(r'Samples:\s*([\d]+)', single_text)
        if samples_match:
            single_samples = int(samples_match.group(1))
            
            # Extract individual timings
            timing_patterns = {
                'single_queue_scan': r'Queue scan:\s+([\d.]+)',
                'single_locking': r'Locking:\s+([\d.]+)',
                'single_grouping': r'Grouping:\s+([\d.]+)',
                'single_dirty_check': r'Dirty check:\s+([\d.]+)',
                'single_array_build': r'Array build:\s+([\d.]+)',
                'single_syscall': r'Syscall:\s+([\d.]+)',
                'single_metadata_update': r'Metadata update:\s+([\d.]+)',
                'single_unlock': r'Unlock:\s+([\d.]+)',
            }
            
            for key, pattern in timing_patterns.items():
                match = re.search(pattern, single_text)
                if match:
                    us_time = float(match.group(1))
                    ns_time = us_time * 1000 * single_samples / thread_count  # Convert us to ns, multiply by samples, divide by thread count
                    timings[key] = ns_time
    
    return timings


def load_time_breakdown_data(timestamps: List[str], base_dir: Path) -> List[Tuple[str, Dict[str, float]]]:
    """
    Load time breakdown data for given timestamps.
    Returns: [(run_id, timings_dict), ...]
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
                
                timings = parse_time_breakdown(output_text)
                
                if timings and '_total_benchmark_time_ns' in timings:
                    data.append((run_id, timings))
    
    return data


def create_time_breakdown_plot(data: List[Tuple[str, Dict[str, float]]], output_dir: Path):
    """
    Create a stacked bar plot showing time breakdown as percentage of total benchmark time.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_ids = []
    breakdown_data = {}
    
    # Components to display
    components = [
        'batch_queue_scan', 'batch_locking', 'batch_grouping', 'batch_dirty_check',
        'batch_array_build', 'batch_syscall', 'batch_metadata_update', 'batch_unlock',
        'single_queue_scan', 'single_locking', 'single_grouping', 'single_dirty_check',
        'single_array_build', 'single_syscall', 'single_metadata_update', 'single_unlock',
    ]
    
    for component in components:
        breakdown_data[component] = []
    breakdown_data['other_unaccounted'] = []
    
    # Collect data for each run
    for run_id, timings in data:
        run_ids.append(run_id)
        total_time = timings.get('_total_benchmark_time_ns', 1)
        thread_count = timings.get('_thread_count', 1)
        
        measured_time = 0
        for component in components:
            component_time = timings.get(component, 0)
            # Each component is already divided by thread_count, so compare to total wall-clock time
            percentage = (component_time / total_time) * 100 if total_time > 0 else 0
            breakdown_data[component].append(percentage)
            measured_time += component_time
        
        # Calculate unaccounted time
        unaccounted_time = total_time - measured_time
        unaccounted_percentage = (unaccounted_time / total_time) * 100 if total_time > 0 else 0
        breakdown_data['other_unaccounted'].append(unaccounted_percentage)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create stacked bar plot
    x_pos = np.arange(len(run_ids))
    width = 0.6
    
    # Colors for different components (use consistent color mapping)
    all_components = components + ['other_unaccounted']
    
    bottom = np.zeros(len(run_ids))
    bars_list = []
    
    for idx, component in enumerate(all_components):
        values = breakdown_data[component]
        label = component if component != 'other_unaccounted' else 'Other/Unaccounted'
        color = COMPONENT_COLORS[component]
        bar = ax.bar(x_pos, values, width, label=label, bottom=bottom, color=color)
        bars_list.append(bar)
        bottom += np.array(values)
    
    # Customize plot
    ax.set_xlabel("Run Configuration", fontsize=12)
    ax.set_ylabel("Time Percentage (%)", fontsize=12)
    ax.set_title("Time Breakdown - Migration Operations (100% = Total Benchmark Time)", fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(run_ids, rotation=45, ha='right')
    ax.set_ylim([0, 100])
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "benchmark_time_breakdown.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close()


def create_migration_only_breakdown_plot(data: List[Tuple[str, Dict[str, float]]], output_dir: Path):
    """
    Create a stacked bar plot showing only migration operation breakdown (excluding Other/Unaccounted).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_ids = []
    breakdown_data = {}
    
    # Components to display (migration operations only)
    components = [
        'batch_queue_scan', 'batch_locking', 'batch_grouping', 'batch_dirty_check',
        'batch_array_build', 'batch_syscall', 'batch_metadata_update', 'batch_unlock',
        'single_queue_scan', 'single_locking', 'single_grouping', 'single_dirty_check',
        'single_array_build', 'single_syscall', 'single_metadata_update', 'single_unlock',
    ]
    
    for component in components:
        breakdown_data[component] = []
    
    # Collect data for each run
    for run_id, timings in data:
        run_ids.append(run_id)
        
        migration_time = 0
        for component in components:
            component_time = timings.get(component, 0)
            migration_time += component_time
        
        # Calculate percentages only within migration operations
        for component in components:
            component_time = timings.get(component, 0)
            percentage = (component_time / migration_time) * 100 if migration_time > 0 else 0
            breakdown_data[component].append(percentage)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create stacked bar plot
    x_pos = np.arange(len(run_ids))
    width = 0.6
    
    # Colors for different components (use consistent color mapping)
    bottom = np.zeros(len(run_ids))
    bars_list = []
    
    for idx, component in enumerate(components):
        values = breakdown_data[component]
        color = COMPONENT_COLORS[component]
        bar = ax.bar(x_pos, values, width, label=component, bottom=bottom, color=color)
        bars_list.append(bar)
        bottom += np.array(values)
    
    # Customize plot
    ax.set_xlabel("Run Configuration", fontsize=12)
    ax.set_ylabel("Time Percentage (%)", fontsize=12)
    ax.set_title("Migration Operations Breakdown (100% = Total Migration Time)", fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(run_ids, rotation=45, ha='right')
    ax.set_ylim([0, 100])
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "benchmark_migration_breakdown.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create a stacked bar plot for time breakdown from benchmark results."
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
    
    print(f"Loading time breakdown data for timestamps: {args.timestamps}")
    data = load_time_breakdown_data(args.timestamps, base_dir)
    
    if not data:
        print("No time breakdown data found. Exiting.")
        return
    
    print(f"Found {len(data)} runs with time breakdown data")
    print(f"Creating stacked bar plots...")
    create_time_breakdown_plot(data, output_dir)
    create_migration_only_breakdown_plot(data, output_dir)
    
    print(f"Done! Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
