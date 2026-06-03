#!/usr/bin/env python3
"""
Paper Figure Generation for SV-Flow

Generates all figures needed for the paper from evaluation CSV/JSON results.

Figures:
  Fig 2: Pareto frontier — QED vs Tanimoto Diversity (scatter + density)
  Fig 3: Spatial diversity boxplots — CentroidVar, PairDist
  Fig 4: Ablation bar chart — 5 variants across 3 metrics
  Fig 5: Physical validity comparison — Clashes, BondAnomalies, BrokenRings

Usage:
    python scripts/plot_results.py \
        --results_json ./results/comparison.json \
        --output_dir ./figures
"""

import argparse
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Try importing matplotlib — gracefully degrade if unavailable
try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print('Warning: matplotlib not available. Install with: pip install matplotlib')


# --- Color scheme ---
COLORS = {
    'DrugFlow':        '#3498db',  # blue
    'SV-Flow Core':    '#e74c3c',  # red
    'SV-Flow FULL':    '#95a5a6',  # gray
    'w/o TP':          '#f39c12',  # orange
    'w/o OP':          '#9b59b6',  # purple
    'Isotropic':       '#1abc9c',  # teal
    'MaxMin':          '#2ecc71',  # green
}

METHOD_LABELS = {
    'drugflow':     'DrugFlow',
    'core':         'SV-Flow Core',
    'full':         'SV-Flow FULL',
    'no_tp':        'w/o Tangent Proj.',
    'no_op':        'w/o Orthogonal Prot.',
    'isotropic':    'Isotropic Repulsion',
    'maxmin':       'DrugFlow N=50 + MaxMin',
}


# ---------------------------------------------------------------------------
# Fig 2: Pareto Frontier (QED vs Tanimoto Diversity)
# ---------------------------------------------------------------------------

def plot_pareto_frontier(results_dict: dict, output_path: str):
    """
    Scatter plot of QED vs Tanimoto Diversity.
    Each method is a point with error bars (std across pockets).
    Also shows the Pareto frontier curve.
    """
    if not HAS_MPL:
        print('Skipping Fig 2: matplotlib not available')
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for method_key, df in results_dict.items():
        if 'qed_mean' not in df.columns or 'tanimoto_diversity' not in df.columns:
            continue

        label = METHOD_LABELS.get(method_key, method_key)
        color = COLORS.get(label, '#333333')

        x_mean = df['tanimoto_diversity'].mean()
        y_mean = df['qed_mean'].mean()
        x_std = df['tanimoto_diversity'].std()
        y_std = df['qed_mean'].std()

        ax.errorbar(x_mean, y_mean, xerr=x_std, yerr=y_std,
                    fmt='o', color=color, label=label,
                    markersize=12, capsize=5, linewidth=2, markeredgewidth=1.5)

    ax.set_xlabel('Tanimoto Diversity (1 − mean similarity)', fontsize=13)
    ax.set_ylabel('QED (Drug-likeness)', fontsize=13)
    ax.set_title('Diversity–Validity Pareto Frontier', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.3)

    # Annotate "better" corner
    ax.annotate('Better ↑', xy=(0.95, 0.95), xycoords='axes fraction',
                fontsize=11, ha='right', va='top', color='gray',
                arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Fig 2 saved to {output_path}')


# ---------------------------------------------------------------------------
# Fig 3: Spatial Diversity Boxplots
# ---------------------------------------------------------------------------

def plot_spatial_diversity(results_dict: dict, output_path: str):
    """Boxplots comparing centroid variance and mean pairwise distance across methods."""
    if not HAS_MPL:
        print('Skipping Fig 3: matplotlib not available')
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metrics = [
        ('centroid_variance', 'Centroid Variance (Å²)'),
        ('mean_pairwise_centroid_distance', 'Mean Pairwise Centroid Distance (Å)'),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):
        data = []
        labels = []
        colors = []

        for method_key, df in results_dict.items():
            if metric not in df.columns:
                continue
            vals = df[metric].dropna().values
            if len(vals) == 0:
                continue

            label = METHOD_LABELS.get(method_key, method_key)
            data.append(vals)
            labels.append(label)
            colors.append(COLORS.get(label, '#333333'))

        positions = range(len(data))
        bp = ax.boxplot(data, positions=positions, patch_artist=True,
                        widths=0.5, showfliers=False)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('Spatial Exploration Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Fig 3 saved to {output_path}')


# ---------------------------------------------------------------------------
# Fig 4: Ablation Bar Chart
# ---------------------------------------------------------------------------

def plot_ablation(results_dict: dict, output_path: str):
    """Grouped bar chart showing 5 variants across diversity, QED, and clashes."""
    if not HAS_MPL:
        print('Skipping Fig 4: matplotlib not available')
        return

    # Only include ablation variants
    variants = ['core', 'full', 'no_tp', 'no_op', 'isotropic']
    metrics = ['tanimoto_diversity', 'qed_mean', 'clashes_per_mol']
    metric_labels = ['Tanimoto Diversity', 'QED', 'Clashes/mol']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric, ylabel in zip(axes, metrics, metric_labels):
        means = []
        stds = []
        labels = []

        for vkey in variants:
            if vkey in results_dict:
                df = results_dict[vkey]
                if metric in df.columns:
                    vals = df[metric].dropna()
                    if len(vals) > 0:
                        means.append(vals.mean())
                        stds.append(vals.std())
                        labels.append(METHOD_LABELS.get(vkey, vkey))

        x = range(len(means))
        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=[COLORS.get(l, '#333') for l in labels],
                      edgecolor='white', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('Ablation Study: Mechanism Contributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Fig 4 saved to {output_path}')


# ---------------------------------------------------------------------------
# Fig 5: Physical Validity Comparison
# ---------------------------------------------------------------------------

def plot_physical_validity(results_dict: dict, output_path: str):
    """Bar chart comparing physical validity metrics across methods."""
    if not HAS_MPL:
        print('Skipping Fig 5: matplotlib not available')
        return

    metrics = ['clashes_per_mol', 'bond_anomaly_rate', 'broken_rings_per_mol']
    metric_labels = ['Clashes/mol', 'Bond Anomaly Rate', 'Broken Rings/mol']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric, ylabel in zip(axes, metrics, metric_labels):
        means = []
        stds = []
        labels = []
        colors_list = []

        for method_key, df in results_dict.items():
            if metric not in df.columns:
                continue
            vals = df[metric].dropna()
            if len(vals) == 0:
                continue

            label = METHOD_LABELS.get(method_key, method_key)
            means.append(vals.mean())
            stds.append(vals.std())
            labels.append(label)
            colors_list.append(COLORS.get(label, '#333333'))

        x = range(len(means))
        ax.bar(x, means, yerr=stds, capsize=5,
               color=colors_list, edgecolor='white', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('Physical Validity Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Fig 5 saved to {output_path}')


# ---------------------------------------------------------------------------
# Combined comparison table (LaTeX-ready)
# ---------------------------------------------------------------------------

def build_comparison_table(results_dict: dict, output_path: str):
    """Build a formatted comparison table with mean ± std for all methods."""
    table_metrics = [
        ('n_valid', 'Valid/Pocket'),
        ('qed_mean', 'QED'),
        ('tanimoto_diversity', 'Tanimoto Div.'),
        ('centroid_variance', 'Centroid Var. (Å²)'),
        ('mean_pairwise_centroid_distance', 'Pair Centroid Dist. (Å)'),
        ('clashes_per_mol', 'Clashes/mol'),
        ('bond_anomaly_rate', 'Bond Anomaly Rate'),
        ('broken_rings_per_mol', 'Broken Rings/mol'),
        ('distance_to_pocket_center', 'Dist. to Pocket (Å)'),
        ('mol_mw', 'MW'),
    ]

    table = {}
    for method_key, df in results_dict.items():
        label = METHOD_LABELS.get(method_key, method_key)
        row = {}
        for col, name in table_metrics:
            if col in df.columns:
                vals = df[col].dropna()
                if len(vals) > 0:
                    row[name] = f'{vals.mean():.3f} ± {vals.std():.3f}'
                else:
                    row[name] = '—'
            else:
                row[name] = '—'
        table[label] = row

    # Print as markdown
    print(f'\n{"="*80}')
    print('Comparison Table')
    print(f'{"="*80}')

    methods = list(table.keys())
    metric_names = [name for _, name in table_metrics]

    # Header
    header = f'| {"Metric":30s} | ' + ' | '.join(f'{m:^20s}' for m in methods) + ' |'
    sep = '|-' + '-'*30 + '-|-' + '-|-'.join('-'*20 for _ in methods) + '-|'
    print(header)
    print(sep)

    for col, name in table_metrics:
        row = f'| {name:30s} | '
        row += ' | '.join(f'{table[m].get(name, "—"):^20s}' for m in methods)
        row += ' |'
        print(row)

    # Save as JSON
    json_path = output_path.replace('.txt', '.json')
    with open(json_path, 'w') as f:
        json.dump(table, f, indent=2)
    print(f'\nTable saved to {json_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_results_from_prefixes(prefixes: dict) -> dict:
    """
    Load evaluation results from multiple CSV prefixes.

    Args:
        prefixes: {method_key: csv_prefix} mapping
    """
    results = {}
    for method_key, prefix in prefixes.items():
        csv_path = f'{prefix}_per_pocket.csv'
        if Path(csv_path).exists():
            results[method_key] = pd.read_csv(csv_path)
            print(f'Loaded {method_key}: {len(results[method_key])} pockets')
        else:
            print(f'Warning: {csv_path} not found, skipping {method_key}')
    return results


def main():
    parser = argparse.ArgumentParser(description='Generate Paper Figures')
    parser.add_argument('--results_json', type=str, default=None,
                        help='JSON mapping method keys to CSV prefixes')
    parser.add_argument('--output_dir', type=str, default='./figures',
                        help='Output directory for figures')
    # Alternative: specify prefixes directly
    parser.add_argument('--drugflow_csv', type=str, default=None)
    parser.add_argument('--core_csv', type=str, default=None)
    parser.add_argument('--full_csv', type=str, default=None)
    parser.add_argument('--maxmin_csv', type=str, default=None)
    parser.add_argument('--isotropic_csv', type=str, default=None)
    args = parser.parse_args()

    # Build prefix mapping
    if args.results_json and Path(args.results_json).exists():
        with open(args.results_json) as f:
            prefixes = json.load(f)
    else:
        prefixes = {}
        if args.drugflow_csv:
            prefixes['drugflow'] = args.drugflow_csv
        if args.core_csv:
            prefixes['core'] = args.core_csv
        if args.full_csv:
            prefixes['full'] = args.full_csv
        if args.maxmin_csv:
            prefixes['maxmin'] = args.maxmin_csv
        if args.isotropic_csv:
            prefixes['isotropic'] = args.isotropic_csv

    if not prefixes:
        print('No input data specified!')
        print('Use --results_json or individual --*_csv arguments')
        return

    # Load all results
    results_dict = load_results_from_prefixes(prefixes)

    if not results_dict:
        print('No results loaded!')
        return

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all figures
    plot_pareto_frontier(results_dict, str(output_dir / 'fig2_pareto_frontier.png'))
    plot_spatial_diversity(results_dict, str(output_dir / 'fig3_spatial_diversity.png'))
    plot_ablation(results_dict, str(output_dir / 'fig4_ablation.png'))
    plot_physical_validity(results_dict, str(output_dir / 'fig5_physical_validity.png'))

    # Build comparison table
    build_comparison_table(results_dict, str(output_dir / 'comparison_table.txt'))

    print(f'\nAll figures saved to {output_dir}/')


if __name__ == '__main__':
    main()
