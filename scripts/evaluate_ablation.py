#!/usr/bin/env python3
"""
消融实验评估脚本

对所有变体 × 口袋组合计算：
  • Chemical Quality:     QED, SA Score, MW, logP, Validity
  • Chemical Diversity:   Tanimoto Diversity (1 - mean pairwise similarity)
  • Spatial Diversity:    Centroid Variance, Mean Pairwise Centroid Distance
  • Physical Validity:    Clashes/mol, Bond Anomaly Rate, Broken Rings/mol

输出:
  • ablation_per_pocket.csv      — 每个 (variant, pocket) 的详细指标
  • ablation_variant_summary.csv — 按 variant 聚合的汇总表
  • ablation_summary.json        — 完整统计 (mean/std/min/max)

用法:
    python scripts/evaluate_ablation.py --input_dir ./output/ablation_core
"""

import argparse
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors
from rdkit.DataStructs import BulkTanimotoSimilarity
from tqdm import tqdm

# Add SVFlow to path
basedir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(basedir))

# Import physical validation
from scripts.validate_physical import (
    parse_pocket_pdb, count_clashes, count_bond_anomalies, count_broken_rings
)


# ── Quality metrics ──────────────────────────────────────────────────────

def compute_qed(mols):
    return [QED.qed(m) for m in mols if m is not None]


def compute_sa_scores(mols):
    try:
        from src.analysis.metrics import MolecularMetrics
        return [MolecularMetrics.calculate_sa(m) for m in mols if m is not None]
    except ImportError:
        return []


def compute_mol_props(mols):
    props = {'mw': [], 'logp': [], 'hba': [], 'hbd': [], 'rotb': []}
    for mol in mols:
        if mol is None:
            continue
        try:
            props['mw'].append(Descriptors.MolWt(mol))
            props['logp'].append(Descriptors.MolLogP(mol))
            props['hba'].append(rdMolDescriptors.CalcNumHBA(mol))
            props['hbd'].append(rdMolDescriptors.CalcNumHBD(mol))
            props['rotb'].append(rdMolDescriptors.CalcNumRotatableBonds(mol))
        except Exception:
            continue
    return {k: np.mean(v) if v else 0.0 for k, v in props.items()}


# ── Diversity metrics ────────────────────────────────────────────────────

def compute_tanimoto_diversity(mols):
    """1 - mean pairwise Tanimoto similarity (Morgan FP, r=2, 2048 bits)."""
    fps = []
    for mol in mols:
        if mol is not None:
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fps.append(fp)
            except Exception:
                continue
    if len(fps) < 2:
        return 0.0
    sims = []
    for i in range(len(fps)):
        sims.extend(BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
    return 1.0 - np.mean(sims) if sims else 0.0


def compute_centroid_variance(mols):
    """Variance of molecular centroids (mean over x/y/z axes)."""
    centroids = []
    for mol in mols:
        if mol is None:
            continue
        try:
            coords = mol.GetConformer().GetPositions()
            centroids.append(coords.mean(axis=0))
        except Exception:
            continue
    if len(centroids) < 2:
        return 0.0
    return float(np.var(np.array(centroids), axis=0).mean())


def compute_mean_pairwise_centroid_distance(mols):
    """Mean pairwise distance between molecular centroids."""
    centroids = []
    for mol in mols:
        if mol is None:
            continue
        try:
            coords = mol.GetConformer().GetPositions()
            centroids.append(coords.mean(axis=0))
        except Exception:
            continue
    if len(centroids) < 2:
        return 0.0
    centroids = np.array(centroids)
    dists = []
    for i in range(len(centroids)):
        d = np.linalg.norm(centroids[i] - centroids[i + 1:], axis=1)
        dists.extend(d.tolist())
    return float(np.mean(dists)) if dists else 0.0


# ── Pocket loading ───────────────────────────────────────────────────────

def load_pocket_coords(pocket_dir: Path):
    """Load pocket atom coordinates for clash detection."""
    pdb_files = list(pocket_dir.glob('*.pdb'))
    if not pdb_files:
        return np.zeros((0, 3)), []
    try:
        pkt = parse_pocket_pdb(str(pdb_files[0]))
        return pkt['coords'], pkt['elements']
    except Exception:
        return np.zeros((0, 3)), []


# ── Main evaluation ──────────────────────────────────────────────────────

def evaluate_variant_pocket(variant_dir: Path, pocket_dir: Path) -> dict:
    """Evaluate all metrics for a single (variant, pocket) combination."""
    pocket_name = pocket_dir.name

    # Load molecules
    mols = []
    for mf in sorted(pocket_dir.glob('mol_*.sdf')):
        supplier = Chem.SDMolSupplier(str(mf), sanitize=True)
        for mol in supplier:
            if mol is not None:
                mols.append(mol)

    n_total = len(mols)
    valid_mols = [m for m in mols if m is not None]
    n_valid = len(valid_mols)

    if n_valid == 0:
        return {
            'pocket': pocket_name,
            'n_total': n_total,
            'n_valid': 0,
        }

    # Quality
    qed_vals = compute_qed(valid_mols)
    sa_vals = compute_sa_scores(valid_mols)
    mol_props = compute_mol_props(valid_mols)

    # Diversity
    tanimoto_div = compute_tanimoto_diversity(valid_mols)
    centroid_var = compute_centroid_variance(valid_mols)
    mean_pair_dist = compute_mean_pairwise_centroid_distance(valid_mols)

    # Physical validity
    pocket_coords, pocket_elements = load_pocket_coords(pocket_dir)
    total_clashes, total_anomalies, total_bonds, total_broken = 0, 0, 0, 0
    for mol in valid_mols:
        nc, _ = count_clashes(mol, pocket_coords, pocket_elements)
        total_clashes += nc
        na, _ = count_bond_anomalies(mol)
        total_anomalies += na
        total_bonds += mol.GetNumBonds()
        total_broken += count_broken_rings(mol)

    return {
        'pocket': pocket_name,
        'n_total': n_total,
        'n_valid': n_valid,
        'qed_mean': np.mean(qed_vals) if qed_vals else 0.0,
        'qed_std': np.std(qed_vals) if qed_vals else 0.0,
        'sa_mean': np.mean(sa_vals) if sa_vals else 0.0,
        'tanimoto_diversity': tanimoto_div,
        'centroid_variance': centroid_var,
        'mean_pairwise_centroid_distance': mean_pair_dist,
        'clashes_per_mol': total_clashes / n_valid if n_valid > 0 else 0.0,
        'bond_anomaly_rate': total_anomalies / total_bonds if total_bonds > 0 else 0.0,
        'broken_rings_per_mol': total_broken / n_valid if n_valid > 0 else 0.0,
        'mw_mean': mol_props.get('mw', 0.0),
        'logp_mean': mol_props.get('logp', 0.0),
        'hba_mean': mol_props.get('hba', 0.0),
        'hbd_mean': mol_props.get('hbd', 0.0),
        'rotb_mean': mol_props.get('rotb', 0.0),
    }


def evaluate_ablation(input_dir: Path):
    """
    Evaluate all (variant, pocket) combinations.
    Expects: input_dir/{variant}/{pocket}/mol_*.sdf
    """
    results = []

    variant_dirs = sorted([d for d in input_dir.iterdir()
                           if d.is_dir() and not d.name.startswith('.')])

    for variant_dir in variant_dirs:
        variant_name = variant_dir.name
        pocket_dirs = sorted([d for d in variant_dir.iterdir()
                              if d.is_dir() and not d.name.startswith('.')])

        print(f'\nEvaluating {variant_name} ({len(pocket_dirs)} pockets)...')
        for pocket_dir in tqdm(pocket_dirs, desc=variant_name):
            row = evaluate_variant_pocket(variant_dir, pocket_dir)
            row['variant'] = variant_name
            results.append(row)

    return pd.DataFrame(results)


def print_variant_summary(df: pd.DataFrame):
    """Print and save aggregated variant comparison table."""
    # ── Aggregate by variant ──
    key_metrics = [
        'n_valid', 'tanimoto_diversity', 'centroid_variance',
        'mean_pairwise_centroid_distance', 'qed_mean', 'sa_mean',
        'clashes_per_mol', 'bond_anomaly_rate', 'broken_rings_per_mol',
    ]

    # Only use metrics that exist
    available = [m for m in key_metrics if m in df.columns]

    summary = df.groupby('variant')[available].agg(['mean', 'std']).round(4)

    print(f'\n{"="*80}')
    print('ABLATION SUMMARY — Per-Variant Aggregates')
    print(f'{"="*80}')
    print(f'{"Variant":15s}', end='')
    for m in available:
        print(f' {m:>22s}', end='')
    print()
    print('-' * (15 + 23 * len(available)))

    variant_order = ['core', 'full', 'wo_op', 'wo_tp', 'isotropic']
    for v in variant_order:
        if v not in df['variant'].values:
            continue
        print(f'{v:15s}', end='')
        for m in available:
            mean_v = summary.loc[v, (m, 'mean')]
            print(f' {mean_v:>22.4f}', end='')
        print()

    print(f'\n{"="*80}')
    print('SPATIAL DIVERSITY vs CORE (centroid_variance ratio)')
    print(f'{"="*80}')

    core_cv = df[df['variant'] == 'core']['centroid_variance'].mean()
    for v in variant_order:
        if v not in df['variant'].values:
            continue
        cv = df[df['variant'] == v]['centroid_variance'].mean()
        ratio = cv / core_cv if core_cv > 0 else 0.0
        print(f'  {v:15s}: {ratio:.2f}x vs Core')

    return summary


def main():
    parser = argparse.ArgumentParser(description='Ablation Study Evaluation')
    parser.add_argument('--input_dir', type=str, default='./output/ablation_core',
                        help='Root directory containing variant subdirectories')
    parser.add_argument('--output_prefix', type=str, default=None,
                        help='Prefix for output files (defaults to input_dir/ablation)')
    args = parser.parse_args()

    warnings.filterwarnings('ignore')

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f'ERROR: Input directory not found: {input_dir}')
        sys.exit(1)

    output_prefix = args.output_prefix or str(input_dir / 'ablation')

    # ── Evaluate ──
    df = evaluate_ablation(input_dir)

    if len(df) == 0:
        print('No data to evaluate!')
        return

    # ── Per-pocket CSV ──
    per_pocket_path = f'{output_prefix}_per_pocket.csv'
    df.to_csv(per_pocket_path, index=False)
    print(f'\nPer-pocket results saved to {per_pocket_path}')

    # ── Variant summary ──
    summary = print_variant_summary(df)
    summary_csv = f'{output_prefix}_variant_summary.csv'
    summary.to_csv(summary_csv)
    print(f'\nVariant summary saved to {summary_csv}')

    # ── Full JSON summary ──
    json_summary = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        vals = df[col].dropna()
        if len(vals) > 0:
            json_summary[col] = {
                'mean': float(vals.mean()),
                'std': float(vals.std()),
                'min': float(vals.min()),
                'max': float(vals.max()),
            }
    json_path = f'{output_prefix}_summary.json'
    with open(json_path, 'w') as f:
        json.dump(json_summary, f, indent=2)
    print(f'Full summary saved to {json_path}')

    # ── LaTeX table (for paper appendix) ──
    latex_path = f'{output_prefix}_table.tex'
    with open(latex_path, 'w') as f:
        f.write(r'\begin{table}[h]' + '\n')
        f.write(r'\caption{Ablation study results on 5 representative pockets.}' + '\n')
        f.write(r'\centering' + '\n')
        f.write(r'\begin{tabular}{lcccc}' + '\n')
        f.write(r'\hline' + '\n')
        f.write(r'Variant & Tanimoto Div. $\uparrow$ & QED $\uparrow$ & '
                r'Clashes/mol $\downarrow$ & Spatial Div. vs Core \\' + '\n')
        f.write(r'\hline' + '\n')

        core_cv = df[df['variant'] == 'core']['centroid_variance'].mean()
        for v in ['core', 'full', 'wo_op', 'wo_tp', 'isotropic']:
            if v not in df['variant'].values:
                continue
            vdf = df[df['variant'] == v]
            td = vdf['tanimoto_diversity'].mean()
            qed = vdf['qed_mean'].mean()
            cl = vdf['clashes_per_mol'].mean()
            cv = vdf['centroid_variance'].mean()
            ratio = cv / core_cv if core_cv > 0 else 0.0

            label_map = {
                'core': 'Core (Ours)',
                'full': 'FULL (TP+OP)',
                'wo_op': 'w/o OP',
                'wo_tp': 'w/o TP',
                'isotropic': 'Isotropic (1/r²)',
            }
            label = label_map.get(v, v)
            f.write(f'{label} & {td:.3f} & {qed:.3f} & '
                    f'{cl:.2f} & {ratio:.2f}x \\\\' + '\n')

        f.write(r'\hline' + '\n')
        f.write(r'\end{tabular}' + '\n')
        f.write(r'\label{tab:ablation}' + '\n')
        f.write(r'\end{table}' + '\n')
    print(f'LaTeX table saved to {latex_path}')


if __name__ == '__main__':
    main()
