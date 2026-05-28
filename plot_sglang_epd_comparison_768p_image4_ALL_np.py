#!/usr/bin/env python3
"""
Script to generate comparison plots for SGLang EPD benchmarks across all configurations.
Compares B70-H200 disaggregation (1E, 2E, 3E, 4E) and H200 aggregation/disaggregation setups with 768p images.
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
OUTPUT_FILE = "/hongming/dynamo/sglang_epd_qwen32_fp8_768p_image4_ALL_np.png"

# Define configurations to compare
configs = {
    "H200 Agg TP1": {
        "path": RESULTS_BASE / "h200_agg_tp1_32b_image4_768p_np_rates",
        "color": "#E74C3C",
        "marker": "o",
        "linestyle": "-"
    },
    "H200 Agg TP2": {
        "path": RESULTS_BASE / "h200_agg_tp2_32b_image4_768p_np_rates",
        "color": "#3498DB",
        "marker": "s",
        "linestyle": "-"
    },
    "H200-H200 Disagg": {
        "path": RESULTS_BASE / "h200_h200_disagg_32b_image4_768p_np_rates",
        "color": "#2ECC71",
        "marker": "^",
        "linestyle": "-"
    },
    "B70-H200 1E+1PD": {
        "path": RESULTS_BASE / "b70_h200_1E_disagg_32b_image4_768p_np_rates",
        "color": "#FF6B6B",
        "marker": "o",
        "linestyle": "--"
    },
    "B70-H200 2E+1PD": {
        "path": RESULTS_BASE / "b70_h200_2E_disagg_32b_image4_768p_np_rates",
        "color": "#4ECDC4",
        "marker": "s",
        "linestyle": "--"
    },
    "B70-H200 3E+1PD": {
        "path": RESULTS_BASE / "b70_h200_3E_disagg_32b_image4_768p_np_rates",
        "color": "#FFE66D",
        "marker": "^",
        "linestyle": "--"
    },
    "B70-H200 4E+1PD": {
        "path": RESULTS_BASE / "b70_h200_4E_disagg_32b_image4_768p_np_rates",
        "color": "#95E1D3",
        "marker": "D",
        "linestyle": "--"
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
    fig, axes = plt.subplots(4, 2, figsize=(18, 24))
    fig.suptitle('EPD Complete Comparison: All Configurations\nQwen3-VL-32B-FP8 | 128 input / 256 output tokens | 4 images Per Request | 768p',
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

        # Plot in reverse order so first configs appear on top when overlapping
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
                   linestyle=config["linestyle"],
                   label=name,
                   alpha=0.9,
                   markeredgewidth=1.5,
                   markeredgecolor='white')

        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(loc='upper left', fontsize=8, framealpha=0.7, ncol=1)
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
