#!/usr/bin/env python3
"""
Drift Verification Experiment

Key question: Is DrugFlow's high centroid variance meaningful exploration
or just molecules drifting away from the binding site?

Experiment:
1. Group DrugFlow molecules by distance to pocket center:
   - Near:  < 5 Å   (within binding site)
   - Mid:   5-10 Å  (peripheral)
   - Far:   > 10 Å  (likely drifted off)
2. Compute per-group metrics: QED, Tanimoto diversity, clash rate
3. Statistical testing: Are Far-group molecules significantly worse?

Expected result:
  Far-group molecules have significantly worse clash rates and lower QED,
  proving DrugFlow's high centroid variance is partly "drift" rather than
  meaningful binding mode exploration.

Usage:
    python scripts/drift_analysis.py \
        --samples_dir ./drugflow_baseline_samples \
        --output ./drift_analysis.csv
"""

import argparse
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED
from rdkit.DataStructs import BulkTanimotoSimilarity
from scipy import stats
from tqdm import tqdm

basedir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(basedir))

DISTANCE_BINS = {
    'Near (< 5Å)':   (0.0, 5.0),
    'Mid (5-10Å)':   (5.0, 10.0),
    'Far (> 10Å)':   (10.0, float('inf')),
}


def load_all_molecules(samples_dir: Path) -> list:
    """Load all molecules with pocket metadata."""
    records = []
    for pocket_dir in sorted(samples_dir.iterdir()):
        if not pocket_dir.is_dir():
            continue
        pocket_name = pocket_dir.name

        # Load pocket PDB to get pocket center
        pocket_files = list(pocket_dir.glob('*.pdb'))
        pocket_center = np.zeros(3)
        if pocket_files:
            try:
                from Bio.PDB import PDBParser
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure('pocket', str(pocket_files[0]))
                coords = []
                for atom in structure.get_atoms():
                    coords.append(atom.get_coord())
                if coords:
                    pocket_center = np.array(coords).mean(axis=0)
            except Exception:
                pass

        # Load molecules
        mol_files = sorted(pocket_dir.glob('mol_*.sdf'))
        for mf in mol_files:
            supplier = Chem.SDMolSupplier(str(mf), sanitize=True)
            for mol in supplier:
                if mol is None:
                    continue
                try:
                    conf = mol.GetConformer()
                    lig_coords = conf.GetPositions()
                    lig_center = lig_coords.mean(axis=0)
                    dist_to_pocket = float(np.linalg.norm(lig_center - pocket_center))

                    records.append({
                        'pocket': pocket_name,
                        'mol': mol,
                        'dist_to_pocket': dist_to_pocket,
                        'n_atoms': mol.GetNumAtoms(),
                        'qed': QED.qed(mol),
                        'mw': Descriptors.MolWt(mol),
                    })
                except Exception:
                    continue

    return records


def compute_morgan_fingerprint(mol):
    """Compute Morgan fingerprint."""
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description='Drift Verification Analysis')
    parser.add_argument('--samples_dir', type=str, required=True,
                        help='Directory of generated molecules')
    parser.add_argument('--output', type=str, default='./drift_analysis.csv',
                        help='Output CSV file')
    args = parser.parse_args()

    warnings.filterwarnings('ignore')

    print('Loading molecules...')
    records = load_all_molecules(Path(args.samples_dir))
    print(f'Loaded {len(records)} molecules')

    if len(records) == 0:
        print('No molecules found!')
        return

    # Group by distance bin
    df = pd.DataFrame(records)
    grouped_stats = []

    for bin_name, (d_min, d_max) in DISTANCE_BINS.items():
        mask = (df['dist_to_pocket'] >= d_min) & (df['dist_to_pocket'] < d_max)
        group = df[mask]
        n = len(group)

        if n == 0:
            grouped_stats.append({
                'distance_bin': bin_name,
                'n_molecules': 0,
                'pct_total': 0.0,
            })
            continue

        # Per-group molecular properties
        stats_row = {
            'distance_bin': bin_name,
            'n_molecules': n,
            'pct_total': 100.0 * n / len(df),
            'qed_mean': group['qed'].mean(),
            'qed_std': group['qed'].std(),
            'mw_mean': group['mw'].mean(),
            'n_atoms_mean': group['n_atoms'].mean(),
            'dist_mean': group['dist_to_pocket'].mean(),
            'dist_std': group['dist_to_pocket'].std(),
        }

        # Chemical diversity within group
        fps = []
        for mol in group['mol']:
            fp = compute_morgan_fingerprint(mol)
            if fp is not None:
                fps.append(fp)

        if len(fps) >= 2:
            similarities = []
            for i in range(min(len(fps), 50)):  # cap at 50 for speed
                sims = BulkTanimotoSimilarity(fps[i], fps[i+1:])
                similarities.extend(sims)
            stats_row['tanimoto_diversity'] = 1.0 - np.mean(similarities) if similarities else 0.0
        else:
            stats_row['tanimoto_diversity'] = 0.0

        grouped_stats.append(stats_row)

    # Print summary
    result_df = pd.DataFrame(grouped_stats)
    print(f'\n{"="*70}')
    print('Drift Analysis Summary')
    print(f'{"="*70}')

    for _, row in result_df.iterrows():
        print(f'\n  {row["distance_bin"]}:')
        print(f'    Molecules: {row["n_molecules"]} ({row["pct_total"]:.1f}%)')
        if row['n_molecules'] > 0:
            print(f'    QED:        {row["qed_mean"]:.3f} ± {row["qed_std"]:.3f}')
            print(f'    Tanimoto:   {row.get("tanimoto_diversity", 0):.3f}')
            print(f'    Mean Dist:  {row["dist_mean"]:.2f} ± {row["dist_std"]:.2f} Å')

    # Statistical tests: Near vs Far
    near_group = df[(df['dist_to_pocket'] >= 0) & (df['dist_to_pocket'] < 5)]
    far_group = df[df['dist_to_pocket'] >= 10]

    if len(near_group) > 5 and len(far_group) > 5:
        print(f'\n{"="*70}')
        print('Statistical Tests: Near (< 5Å) vs Far (> 10Å)')
        print(f'{"="*70}')

        # QED
        t_stat, p_val = stats.ttest_ind(near_group['qed'], far_group['qed'], equal_var=False)
        print(f'  QED:     t={t_stat:.3f}, p={p_val:.4f}  {"*" if p_val < 0.05 else "NS"}')

        # Distance (sanity check — should be significant)
        t_stat, p_val = stats.ttest_ind(near_group['dist_to_pocket'], far_group['dist_to_pocket'], equal_var=False)
        print(f'  Dist:    t={t_stat:.3f}, p={p_val:.4f}  {"*" if p_val < 0.05 else "NS"}')

        # Effect size (Cohen's d)
        d_qed = (near_group['qed'].mean() - far_group['qed'].mean()) / \
                np.sqrt((near_group['qed'].var() + far_group['qed'].var()) / 2)
        print(f'  Cohen d (QED): {d_qed:.3f}')

    # Save
    result_df.to_csv(args.output, index=False)
    print(f'\nResults saved to {args.output}')


if __name__ == '__main__':
    main()
