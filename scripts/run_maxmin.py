#!/usr/bin/env python3
"""
MaxMin Diversity Selection Baseline (P0-2)

Generates N=50 molecules per pocket using DrugFlow independent sampling,
then uses RDKit's MaxMinPicker to select the K=10 most chemically diverse
molecules based on Tanimoto distance.

This baseline addresses the key question: "Is SV-Flow's diversity simply
due to sampling more molecules, or does the coupled SVGD repulsion produce
fundamentally different exploration behavior?"

Comparison logic:
  - DrugFlow N=10:     independent, no post-processing
  - DrugFlow N=50+MM:  independent + post-hoc diversity selection
  - SV-Flow Core N=10: coupled SVGD during sampling

If SV-Flow Core (N=10) matches or exceeds N=50+MaxMin, it proves the
superiority of coupled repulsion over post-hoc filtering.

Usage:
    # Step 1: Generate N=50 molecules (or use pre-generated)
    python scripts/generate_baseline.py \
        --n_samples 50 --output_dir ./drugflow_n50_samples

    # Step 2: Run MaxMin selection
    python scripts/run_maxmin.py \
        --samples_dir ./drugflow_n50_samples \
        --k 10 \
        --output_dir ./maxmin_selected
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.SimDivFilters import MaxMinPicker
from tqdm import tqdm

basedir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(basedir))


def compute_fingerprints(mols):
    """Compute Morgan fingerprints for a list of RDKit molecules."""
    fps = []
    valid_indices = []
    for i, mol in enumerate(mols):
        if mol is not None:
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fps.append(fp)
                valid_indices.append(i)
            except Exception:
                continue
    return fps, valid_indices


def maxmin_select(mols, k=10):
    """
    Select K most diverse molecules using MaxMin algorithm on Tanimoto distance.

    Args:
        mols: list of RDKit molecules
        k: number of molecules to select

    Returns:
        selected_indices: indices of selected molecules in original list
        selected_mols: the selected RDKit molecules
    """
    if len(mols) <= k:
        return list(range(len(mols))), mols

    fps, valid_indices = compute_fingerprints(mols)
    if len(fps) <= k:
        return valid_indices, [mols[i] for i in valid_indices]

    # MaxMinPicker: starts with a random molecule, then iteratively picks
    # the molecule with maximum minimum Tanimoto distance to the selected set.
    n_fps = len(fps)

    def dist_func(i, j):
        return 1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j])

    picker = MaxMinPicker()
    selected_fp_indices = picker.LazyBitVectorPick(
        lambda i, j: dist_func(i, j),
        poolSize=n_fps,
        pickSize=k,
        seed=42,
    )

    selected_indices = [valid_indices[i] for i in selected_fp_indices]
    selected_mols = [mols[i] for i in selected_indices]
    return selected_indices, selected_mols


def compute_pairwise_tanimoto_diversity(mols):
    """Compute average pairwise Tanimoto diversity (1 - mean similarity)."""
    fps, _ = compute_fingerprints(mols)
    if len(fps) < 2:
        return 0.0

    from rdkit.DataStructs import BulkTanimotoSimilarity
    similarities = []
    for i in range(len(fps)):
        sims = BulkTanimotoSimilarity(fps[i], fps[i+1:])
        similarities.extend(sims)

    return 1.0 - np.mean(similarities) if similarities else 0.0


def main():
    parser = argparse.ArgumentParser(description='MaxMin Diversity Selection')
    parser.add_argument('--samples_dir', type=str, required=True,
                        help='Directory containing DrugFlow N=50 samples')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of molecules to select')
    parser.add_argument('--output_dir', type=str, default='./maxmin_selected',
                        help='Output directory for selected molecules')
    args = parser.parse_args()

    warnings.filterwarnings('ignore')

    samples_dir = Path(args.samples_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pocket_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir()])
    print(f'Processing {len(pocket_dirs)} pocket directories...')

    all_results = []
    for pocket_dir in tqdm(pocket_dirs, desc='MaxMin selection'):
        pocket_name = pocket_dir.name

        # Load all molecules
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

        # Compute full-set diversity
        full_diversity = compute_pairwise_tanimoto_diversity(mols)

        # MaxMin selection
        selected_indices, selected_mols = maxmin_select(mols, k=args.k)
        selected_diversity = compute_pairwise_tanimoto_diversity(selected_mols)

        # Save selected molecules
        pocket_outdir = output_dir / pocket_name
        pocket_outdir.mkdir(parents=True, exist_ok=True)
        for i, mol in enumerate(selected_mols):
            if mol is not None:
                out_sdf = pocket_outdir / f'mol_{i:02d}.sdf'
                Chem.MolToMolFile(mol, str(out_sdf))

        # Copy pocket PDB
        pocket_files = list(pocket_dir.glob('*.pdb'))
        if pocket_files:
            import shutil
            shutil.copy(str(pocket_files[0]), str(pocket_outdir / 'pocket.pdb'))

        all_results.append({
            'pocket': pocket_name,
            'n_available': len(mols),
            'n_selected': len(selected_mols),
            'full_diversity': full_diversity,
            'selected_diversity': selected_diversity,
        })

    # Summary
    if all_results:
        diversities = [r['selected_diversity'] for r in all_results]
        full_diversities = [r['full_diversity'] for r in all_results]
        print(f'\n{"="*60}')
        print(f'MaxMin Selection Summary ({len(all_results)} pockets)')
        print(f'{"="*60}')
        print(f'  Full pool diversity (N={args.k}):     {np.mean(full_diversities):.4f}')
        print(f'  Post-MaxMin diversity (K={args.k}):   {np.mean(diversities):.4f}')
        print(f'  Diversity gain from selection:        {np.mean(diversities) - np.mean(full_diversities):.4f}')

        import json
        summary_path = output_dir / 'maxmin_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    print(f'\nSelected molecules saved to {output_dir}')


if __name__ == '__main__':
    main()
