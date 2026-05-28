#!/usr/bin/env python3
"""
Script to generate comparison plots for SGLang EPD benchmarks across different configurations.
Based on the reference plot structure from epd_qwen32_fp8_128_256_20_images_480p_all.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
plt.style.use('dark_background')
sns.set_palette("husl")

# Configuration
RESULTS_BASE = Path("/hongming/res")
OUTPUT_FILE = "/hongming/dynamo/sglang_epd_qwen32_fp8_128_256_8_images_1080p_all.png"

# Define configurations to compare
configs = {
    "H200 Agg TP1": {
        "path": RESULTS_BASE / "agg_1080p_tp1_h200_32b_image8",
        "color": "#FF6B6B",
        "marker": "o"
    },
    "H200 Agg TP2": {
        "path": RESULTS_BASE / "agg_1080p_tp2_h200_32b_image8",
        "color": "#4ECDC4",
        "marker": "s"
    },
    "H200 Disagg": {
        "path": RESULTS_BASE / "disagg_1080p_h200_h200_32b_image8_n32",
        "color": "#FFE66D",
        "marker": "^"
    }
}

def load_results(config_path):
    """Load results_summary.csv from the most recent test directory"""
    # Find the test directory (should be only one)
    test_dirs = list(config_path.glob("test_sglang_multi_rates_*"))
    if not test_dirs:
        raise FileNotFoundError(f"No test directories found in {config_path}")

    test_dir = test_dirs[0]  # Use the first (and likely only) test directory
    csv_path = test_dir / "results_summary.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"results_summary.csv not found in {test_dir}")

    df = pd.read_csv(csv_path)
    return df

def create_plot():
    """Create the 8-panel comparison plot"""
    fig, axes = plt.subplots(4, 2, figsize=(16, 24))
    fig.suptitle('EPD Disaggregation\nQwen3-VL-32B-FP8 | 128 input / 256 output tokens | 8 images Per Request | 1080p',
                 fontsize=16, fontweight='bold', y=0.995)

    # Load data for all configs
    data = {}
    for name, config in configs.items():
        try:
            data[name] = load_results(config["path"])
            print(f"Loaded data for {name}: {len(data[name])} rows")
        except Exception as e:
            print(f"Warning: Could not load {name}: {e}")
            continue

    if not data:
        raise ValueError("No data could be loaded from any configuration")

    # Define metrics to plot
    metrics = [
        ("median_ttft_ms", "Median TTFT vs Request Rate", "Request Rate (req/s)", "Median TTFT (ms)"),
        ("p99_ttft_ms", "P99 TTFT vs Request Rate", "Request Rate (req/s)", "P99 TTFT (ms)"),
        ("median_tpot_ms", "Median TPOT vs Request Rate", "Request Rate (req/s)", "Median TPOT (ms)"),
        ("p99_tpot_ms", "P99 TPOT vs Request Rate", "Request Rate (req/s)", "P99 TPOT (ms)"),
        ("median_itl_ms", "Median ITL vs Request Rate", "Request Rate (req/s)", "Median ITL (ms)"),
        ("p99_itl_ms", "P99 ITL vs Request Rate", "Request Rate (req/s)", "P99 ITL (ms)"),
        ("actual_rps", "Request Throughput vs Request Rate", "Request Rate (req/s)", "Request Throughput (req/s)"),
        ("output_throughput_toks", "Output Token Throughput vs Request Rate", "Request Rate (req/s)", "Output Token Throughput (tok/s)")
    ]

    # Plot each metric
    for idx, (metric, title, xlabel, ylabel) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        # Plot in reverse order so TP1 appears on top when overlapping
        for name in reversed(list(data.keys())):
            df = data[name]
            config = configs[name]

            # Use target_rate for x-axis
            x = df['target_rate']
            y = df[metric]

            # Plot line and markers with thicker lines for better visibility
            ax.plot(x, y,
                   color=config["color"],
                   marker=config["marker"],
                   markersize=10,
                   linewidth=2.5,
                   linestyle='--',
                   label=name,
                   alpha=0.9,
                   markeredgewidth=1.5,
                   markeredgecolor='white')

        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.7)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Set x-axis to log scale if appropriate
        if idx < 6:  # TTFT, TPOT, and ITL metrics
            ax.set_xscale('log')
            ax.set_xticks([0.1, 0.25, 0.5, 1.0])
            ax.set_xticklabels(['0.1', '0.25', '0.5', '1.0'])

        # Format y-axis
        ax.ticklabel_format(style='plain', axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    print(f"\nPlot saved to: {OUTPUT_FILE}")

    # Print summary statistics
    print("\n" + "="*80)
    print("Summary Statistics:")
    print("="*80)
    for name, df in data.items():
        print(f"\n{name}:")
        print(f"  Request rates tested: {df['target_rate'].tolist()}")
        print(f"  Median TTFT range: {df['median_ttft_ms'].min():.1f} - {df['median_ttft_ms'].max():.1f} ms")
        print(f"  P99 TTFT range: {df['p99_ttft_ms'].min():.1f} - {df['p99_ttft_ms'].max():.1f} ms")
        print(f"  Max output throughput: {df['output_throughput_toks'].max():.1f} tok/s")

if __name__ == "__main__":
    create_plot()
