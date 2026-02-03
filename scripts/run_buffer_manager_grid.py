#!/usr/bin/env python3

import argparse
import itertools
import json
import os
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep buffer manager config values and run hyriseBenchmarkBufferManager."
    )
    parser.add_argument(
        "--config",
        default="/users/yrayhan/hyrise/buffer_manager_config.json",
        help="Path to buffer manager config JSON.",
    )
    parser.add_argument(
        "--benchmark-bin",
        default="/users/yrayhan/hyrise/cmake-build-debug/hyriseBenchmarkBufferManager",
        help="Path to benchmark binary.",
    )
    parser.add_argument(
        "--benchmark-filter",
        default="BM_ycsb/UpdateHeavy/CustomMigrationPolicy/16/iterations:1/repeats:1/real_time/threads:11",
        help="Google benchmark filter string.",
    )
    parser.add_argument(
        "--output-dir",
        default="/users/yrayhan/hyrise/benchmark_results",
        help="Directory to store per-run outputs.",
    )
    return parser.parse_args()


def run_benchmark(config_path: Path, benchmark_bin: Path, benchmark_filter: str):
    cmd = [
        "numactl",
        "--cpunodebind=0",
        "env",
        f"HYRISE_BUFFER_MANAGER_CONFIG_JSON_PATH={config_path}",
        str(benchmark_bin),
        f"--benchmark_filter={benchmark_filter}",
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    args = parse_args()

    config_path = Path(args.config)
    benchmark_bin = Path(args.benchmark_bin)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with config_path.open("r", encoding="utf-8") as f:
        original_config = json.load(f)

    # Define sweep values here
    batch_sizes = [32, 64, 128, 256, 512]
    migration_modes = [0, 1, 2]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = output_dir / f"summary_{timestamp}.jsonl"

    try:
        for use_custom_syscall in [True, False]:
            # When use_custom_syscall is True, only use migration_mode 0
            modes_to_use = [0] if use_custom_syscall else migration_modes
            
            for batch_size in batch_sizes:
                for migration_mode in modes_to_use:
                    # min_demotion_batch_size, min_promotion_batch_size, and migration_max_bs all get the same value
                    min_demotion = batch_size
                    min_promotion = batch_size
                    migration_max_bs = batch_size

                    current_config = deepcopy(original_config)
                    current_config["use_custom_syscall"] = use_custom_syscall
                    current_config["min_demotion_batch_size"] = min_demotion
                    current_config["min_promotion_batch_size"] = min_promotion
                    current_config["migration_mode"] = migration_mode
                    current_config["migration_max_bs"] = migration_max_bs

                    with config_path.open("w", encoding="utf-8") as f:
                        json.dump(current_config, f, indent=2)
                        f.write("\n")

                    result = run_benchmark(config_path, benchmark_bin, args.benchmark_filter)

                    run_id = (
                        f"ucs-{int(use_custom_syscall)}_"
                        f"mindem-{min_demotion}_"
                        f"minpro-{min_promotion}_"
                        f"mm-{migration_mode}_"
                        f"mmbs-{migration_max_bs}"
                    )

                    output_path = output_dir / f"run_{timestamp}_{run_id}.txt"
                    with output_path.open("w", encoding="utf-8") as out:
                        out.write("CONFIG\n")
                        out.write(json.dumps(current_config, indent=2))
                        out.write("\n\nSTDOUT\n")
                        out.write(result.stdout)
                        out.write("\n\nSTDERR\n")
                        out.write(result.stderr)

                    summary_record = {
                        "run_id": run_id,
                        "config": current_config,
                        "return_code": result.returncode,
                        "output_file": str(output_path),
                    }
                    with summary_path.open("a", encoding="utf-8") as summary:
                        summary.write(json.dumps(summary_record))
                        summary.write("\n")

        # Single run with enable_batch_eviction and enable_batch_promotion disabled
        print("Running benchmark with enable_batch_eviction=false and enable_batch_promotion=false...")
        current_config = deepcopy(original_config)
        current_config["enable_batch_eviction"] = False
        current_config["enable_batch_promotion"] = False

        with config_path.open("w", encoding="utf-8") as f:
            json.dump(current_config, f, indent=2)
            f.write("\n")

        result = run_benchmark(config_path, benchmark_bin, args.benchmark_filter)

        run_id = "nobatch"

        output_path = output_dir / f"run_{timestamp}_{run_id}.txt"
        with output_path.open("w", encoding="utf-8") as out:
            out.write("CONFIG\n")
            out.write(json.dumps(current_config, indent=2))
            out.write("\n\nSTDOUT\n")
            out.write(result.stdout)
            out.write("\n\nSTDERR\n")
            out.write(result.stderr)

        summary_record = {
            "run_id": run_id,
            "config": current_config,
            "return_code": result.returncode,
            "output_file": str(output_path),
        }
        with summary_path.open("a", encoding="utf-8") as summary:
            summary.write(json.dumps(summary_record))
            summary.write("\n")
    finally:
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(original_config, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
