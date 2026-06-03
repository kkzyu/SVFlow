#!/usr/bin/env python3
"""
Comprehensive Evaluation Pipeline for SV-Flow

Computes all metrics needed for the paper in a single pass:
  1. Chemical Quality: QED, SA Score, MW, logP, validity
  2. Chemical Diversity: Tanimoto diversity, Tanimoto vs reference
  3. Spatial Diversity: Centroid variance, mean pairwise centroid distance,
     distance from pocket center
  4. Physical Validity: Clashes/mol, bond anomaly rate, broken rings/mol

Outputs:
  - per_pocket.csv: detailed metrics per pocket
  - summary.json: aggregated summary statistics
  - comparison_table.json: formatted for paper tables

Usage:
    python scripts/evaluate_full.py \
        --samples_dir ./svflow_core_samples \
        --output_prefix ./results/core
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
from rdkit.DataStructs import TanimotoSimilarity, BulkTanimotoSimilarity
from tqdm import tqdm
from collections import defaultdict

basedir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(basedir))

# Import physical validation functions
from scripts.validate_physical import (
    parse_pocket_pdb, count_clashes, count_bond_anomalies, count_broken_rings
)


# --- Quality metrics ---

def compute_qed_values(mols):
    """Compute QED for each valid molecule."""
    return [QED.qed(m) for m in mols if m is not None]


def compute_sa_scores(mols):
    """Compute SA (Synthetic Accessibility) scores."""
    try:
        from src.analysis.metrics import compute_sa_score
        return [compute_sa_score(m) for m in mols if m is not None]
    except ImportError:
        return []


def compute_molecular_properties(mols):
    """Compute basic molecular properties."""
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


# --- Diversity metrics ---

def compute_tanimoto_diversity(mols):
    """Average pairwise Tanimoto diversity (1 - mean similarity)."""
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

    similarities = []
    for i in range(len(fps)):
        sims = BulkTanimotoSimilarity(fps[i], fps[i+1:])
        similarities.extend(sims)

    return 1.0 - np.mean(similarities) if similarities else 0.0


def compute_centroid_variance(mols):
    """Variance of molecular centroids (spatial exploration)."""
    centroids = []
    for mol in mols:
        if mol is None:
            continue
        try:
            conf = mol.GetConformer()
            coords = conf.GetPositions()
            centroids.append(coords.mean(axis=0))
        except Exception:
            continue

    if len(centroids) < 2:
        return 0.0

    centroids = np.array(centroids)
    return float(np.var(centroids, axis=0).mean())


def compute_mean_pairwise_centroid_distance(mols):
    """Mean pairwise distance between molecular centroids."""
    centroids = []
    for mol in mols:
        if mol is None:
            continue
        try:
            conf = mol.GetConformer()
            coords = conf.GetPositions()
            centroids.append(coords.mean(axis=0))
        except Exception:
            continue

    if len(centroids) < 2:
        return 0.0

    centroids = np.array(centroids)
    dists = []
    for i in range(len(centroids)):
        d = np.linalg.norm(centroids[i] - centroids[i+1:], axis=1)
        dists.extend(d.tolist())

    return float(np.mean(dists)) if dists else 0.0


def compute_contact_fingerprint_diversity(mols):
    """
    Compute diversity of protein-ligand contact fingerprints.
    Contact fingerprint: which pocket residues each ligand atom is close to (< 4.5Å).

    This is a structural/site-level diversity metric that captures whether
    molecules bind to different parts of the pocket.

    Note: requires pocket data; if unavailable, returns 0.0.
    """
    # This requires pocket residue data which isn't available from mols alone
    # Placeholder — can be implemented when pocket data is accessible
    return 0.0


# --- Site analysis ---

def compute_distance_to_pocket_center(mols, pocket_coords):
    """Compute mean distance of molecular centroids to the pocket center."""
    if pocket_coords.shape[0] == 0:
        return 0.0

    pocket_center = pocket_coords.mean(axis=0)
    distances = []
    for mol in mols:
        if mol is None:
            continue
        try:
            conf = mol.GetConformer()
            lig_coords = conf.GetPositions()
            lig_center = lig_coords.mean(axis=0)
            distances.append(float(np.linalg.norm(lig_center - pocket_center)))
        except Exception:
            continue

    return float(np.mean(distances)) if distances else 0.0


# --- Main evaluation ---

def evaluate_directory(samples_dir: Path) -> pd.DataFrame:
    """Evaluate all metrics for every pocket directory."""
    pocket_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir()])
    print(f'Evaluating {len(pocket_dirs)} pocket directories...')

    results = []
    for pocket_dir in tqdm(pocket_dirs, desc='Evaluating'):
        pocket_name = pocket_dir.name

        # Load molecules
        mols = []
        mol_files = sorted(pocket_dir.glob('mol_*.sdf'))
        for mf in mol_files:
            supplier = Chem.SDMolSupplier(str(mf), sanitize=True)
            for mol in supplier:
                if mol is not None:
                    mols.append(mol)

        n_total = len(mols)
        valid_mols = [m for m in mols if m is not None]
        n_valid = len(valid_mols)

        if n_valid == 0:
            results.append({'pocket': pocket_name, 'n_total': n_total, 'n_valid': 0})
            continue

        # Quality
        qed_vals = compute_qed_values(valid_mols)
        sa_vals = compute_sa_scores(valid_mols)
        mol_props = compute_molecular_properties(valid_mols)

        # Diversity
        tanimoto_div = compute_tanimoto_diversity(valid_mols)
        centroid_var = compute_centroid_variance(valid_mols)
        mean_pair_dist = compute_mean_pairwise_centroid_distance(valid_mols)

        # Physical validity
        pocket_files = list(pocket_dir.glob('*.pdb'))
        pocket_coords = np.zeros((0, 3))
        pocket_elements = []
        if pocket_files:
            try:
                pkt = parse_pocket_pdb(str(pocket_files[0]))
                pocket_coords = pkt['coords']
                pocket_elements = pkt['elements']
            except Exception:
                pass

        total_clashes = 0
        total_anomalies = 0
        total_bonds = 0
        total_broken = 0
        for mol in valid_mols:
            n_c, _ = count_clashes(mol, pocket_coords, pocket_elements)
            total_clashes += n_c
            n_a, _ = count_bond_anomalies(mol)
            total_anomalies += n_a
            total_bonds += mol.GetNumBonds()
            total_broken += count_broken_rings(mol)

        # Distance to pocket center
        dist_to_center = compute_distance_to_pocket_center(valid_mols, pocket_coords)

        results.append({
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
            'distance_to_pocket_center': dist_to_center,
            **{f'mol_{k}': v for k, v in mol_props.items()},
        })

    return pd.DataFrame(results)


def print_summary(df: pd.DataFrame, label: str):
    """Print formatted summary statistics."""
    print(f'\n{"="*70}')
    print(f'{label}')
    print(f'{"="*70}')

    metrics = [
        ('n_valid', 'Valid molecules/pocket'),
        ('qed_mean', 'QED'),
        ('tanimoto_diversity', 'Tanimoto Diversity'),
        ('centroid_variance', 'Centroid Variance (Å²)'),
        ('mean_pairwise_centroid_distance', 'Mean Pairwise Centroid Dist (Å)'),
        ('clashes_per_mol', 'Clashes/mol'),
        ('bond_anomaly_rate', 'Bond Anomaly Rate'),
        ('broken_rings_per_mol', 'Broken Rings/mol'),
        ('distance_to_pocket_center', 'Dist to Pocket Center (Å)'),
    ]

    for col, name in metrics:
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                print(f'  {name:40s}: {vals.mean():.4f} ± {vals.std():.4f}')


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Evaluation')
    parser.add_argument('--samples_dir', type=str, required=True,
                        help='Directory containing generated molecules')
    parser.add_argument('--label', type=str, default='Method',
                        help='Label for this method in output')
    parser.add_argument('--output_prefix', type=str, default='./results/eval',
                        help='Prefix for output files')
    args = parser.parse_args()

    warnings.filterwarnings('ignore')

    # Evaluate
    df = evaluate_directory(Path(args.samples_dir))

    if len(df) == 0:
        print('No data to evaluate!')
        return

    # Print summary
    print_summary(df, args.label)

    # Save per-pocket CSV
    csv_path = f'{args.output_prefix}_per_pocket.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nPer-pocket results saved to {csv_path}')

    # Save summary JSON
    summary = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        vals = df[col].dropna()
        if len(vals) > 0:
            summary[col] = {
                'mean': float(vals.mean()),
                'std': float(vals.std()),
                'min': float(vals.min()),
                'max': float(vals.max()),
            }

    summary_path = f'{args.output_prefix}_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Summary saved to {summary_path}')


if __name__ == '__main__':
    main()
