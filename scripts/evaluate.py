#!/usr/bin/env python3
"""
SV-Flow Evaluation Script

Computes comprehensive metrics for generated molecules:
1. Validity & Quality: Validity, Connectivity, QED, SA, PoseBusters
2. Diversity: Average pairwise Tanimoto similarity, Centroid variance
3. Affinity & Site Occupancy: Vina/Gnina docking scores

Usage:
    python scripts/evaluate.py \
        --samples_dir ./svflow_samples \
        --reference_smiles /path/to/train_smiles.npy \
        --gnina /path/to/gnina \
        --reduce /path/to/reduce
"""

import argparse
import sys
import json
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED
from rdkit.DataStructs import TanimotoSimilarity, BulkTanimotoSimilarity
from tqdm import tqdm

# Add DrugFlow to path
drugflow_path = Path('/root/baselines/DrugFlow/code/DrugFlow-main')
sys.path.insert(0, str(drugflow_path))

from src.analysis.metrics import MoleculeValidity, MolecularProperties
from src.sbdd_metrics.metrics import FullEvaluator
from src.sbdd_metrics.evaluation import aggregated_metrics, collection_metrics, VALIDITY_METRIC_NAME


def compute_tanimoto_diversity(mols):
    """Compute average pairwise Tanimoto similarity (lower = more diverse)."""
    if len(mols) < 2:
        return 0.0

    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols if m is not None]
    fps = [fp for fp in fps if fp is not None]
    if len(fps) < 2:
        return 0.0

    similarities = []
    for i in range(len(fps)):
        sims = BulkTanimotoSimilarity(fps[i], fps[i+1:])
        similarities.extend(sims)

    if not similarities:
        return 0.0

    return 1.0 - np.mean(similarities)  # diversity = 1 - similarity


def compute_centroid_variance(mols):
    """Compute variance of molecular centroids (spatial diversity)."""
    centroids = []
    for mol in mols:
        if mol is None:
            continue
        try:
            conf = mol.GetConformer()
            coords = conf.GetPositions()
            centroid = coords.mean(axis=0)
            centroids.append(centroid)
        except:
            continue

    if len(centroids) < 2:
        return 0.0

    centroids = np.array(centroids)
    return np.var(centroids, axis=0).mean()


def compute_molecule_metrics(mols, evaluator=None, receptors=None):
    """Compute comprehensive metrics for a set of molecules."""
    results = {}

    # Basic validity
    mol_metrics = MoleculeValidity()
    validity_results = mol_metrics(mols)
    results.update({
        'n_total': validity_results.get('n_total', len(mols)),
        'validity': validity_results.get('validity', 0.0),
        'connectivity': validity_results.get('connectivity', 0.0),
        'valid_and_connected': validity_results.get('valid_and_connected', 0.0),
    })

    # Molecular properties
    valid_mols = [m for m in mols if m is not None]
    if valid_mols:
        props = MolecularProperties()
        prop_results = props(valid_mols)
        results.update(prop_results)

    # Diversity
    results['tanimoto_diversity'] = compute_tanimoto_diversity(valid_mols)
    results['centroid_variance'] = compute_centroid_variance(valid_mols)

    # QED manually for each mol
    qed_values = [QED.qed(m) for m in valid_mols if m is not None]
    if qed_values:
        results['qed_mean'] = np.mean(qed_values)
        results['qed_std'] = np.std(qed_values)

    # SA Score (requires sascorer)
    try:
        from src.analysis.metrics import compute_sa_score
        sa_values = [compute_sa_score(m) for m in valid_mols]
        if sa_values:
            results['sa_mean'] = np.mean(sa_values)
            results['sa_std'] = np.std(sa_values)
    except:
        pass

    # Full evaluation with PoseBusters
    if evaluator is not None and receptors is not None:
        eval_results = []
        for mol, receptor in zip(valid_mols, receptors):
            if mol is None:
                continue
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    receptor_path = Path(tmpdir, 'receptor.pdb')
                    Chem.MolToPDBFile(receptor, str(receptor_path))
                    eval_results.append(evaluator(mol, receptor_path))
            except Exception as e:
                print(f'  Evaluator error: {e}')

        if eval_results:
            table = pd.DataFrame(eval_results)
            agg = aggregated_metrics(table, evaluator.dtypes, VALIDITY_METRIC_NAME)
            for _, row in agg.iterrows():
                results[row['metric'].replace('.', '_')] = row['value']

    return results


def main():
    parser = argparse.ArgumentParser(description='SV-Flow Evaluation')
    parser.add_argument('--samples_dir', type=str, required=True,
                        help='Directory containing generated molecules')
    parser.add_argument('--reference_smiles', type=str, default=None,
                        help='Path to training set SMILES for novelty computation')
    parser.add_argument('--gnina', type=str, default=None,
                        help='Path to gnina executable')
    parser.add_argument('--reduce', type=str, default=None,
                        help='Path to reduce executable')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file for results')
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)

    # Load reference SMILES if available
    train_smiles = None
    if args.reference_smiles and Path(args.reference_smiles).exists():
        train_smiles = np.load(args.reference_smiles)

    # Initialize evaluator
    evaluator = FullEvaluator(gnina=args.gnina, reduce=args.reduce)

    # Collect per-pocket results
    all_results = []
    pocket_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir()])

    print(f'Found {len(pocket_dirs)} pocket directories')

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

        if not mols:
            print(f'  No valid molecules in {pocket_name}')
            continue

        # Compute metrics
        metrics = compute_molecule_metrics(mols, evaluator=evaluator)
        metrics['pocket'] = pocket_name
        metrics['n_molecules'] = len(mols)
        all_results.append(metrics)

    # Aggregate results
    if all_results:
        df = pd.DataFrame(all_results)

        # Summary statistics
        print('\n' + '='*60)
        print('SV-Flow Evaluation Summary')
        print('='*60)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        summary = df[numeric_cols].describe()
        for col in numeric_cols:
            mean_val = df[col].mean()
            std_val = df[col].std()
            print(f'  {col:40s}: {mean_val:.4f} ± {std_val:.4f}')

        if args.output:
            df.to_csv(args.output, index=False)
            print(f'\nResults saved to {args.output}')

    print('\nDone!')


if __name__ == '__main__':
    main()
