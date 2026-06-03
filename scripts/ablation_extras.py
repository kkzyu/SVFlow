#!/usr/bin/env python3
"""
Supplementary Ablation Experiments

Two experiments in one script:

1. No-Annealing Ablation (t_on=1.0 vs t_on=0.5):
   Purpose: Prove time-annealed scheduling is necessary.
   Running SVGD from the very beginning (t=1.0 noise state) should
   introduce artifacts: reduced validity, higher bond anomalies.

2. N-Trajectory Scalability (N ∈ {2, 4, 8, 16, 32}):
   Purpose: Show spatial diversity scales with trajectory count.
   Expected: diversity gains peak at N=4-8, then diminish due to
   finite pocket volume.

Usage:
    # No-annealing ablation
    python scripts/ablation_extras.py \
        --checkpoint /root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt \
        --mode no_annealing \
        --max_pockets 10

    # Scalability
    python scripts/ablation_extras.py \
        --checkpoint /root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt \
        --mode scalability \
        --max_pockets 3
"""

import argparse
import sys
import json
import time
import warnings
from pathlib import Path

import torch
import numpy as np

# Fix for PyTorch 2.6+
torch.serialization.add_safe_globals([argparse.Namespace])

basedir = Path(__file__).resolve().parent.parent
drugflow_path = Path('/root/baselines/DrugFlow/code/DrugFlow-main')
sys.path.insert(0, str(drugflow_path))
sys.path.insert(0, str(basedir))

from src import utils
from src.model.lightning import DrugFlow
from src.data.data_utils import process_raw_pair, TensorDict, Residues
from Bio.PDB import PDBParser
from rdkit import Chem
from tqdm import tqdm

from svflow.sampler import SVFlowSampler


def find_test_pockets(test_dir, max_pockets=None):
    pockets = []
    for pdb_file in sorted(test_dir.rglob('*.pdb')):
        sdf_file = pdb_file.with_suffix('.sdf')
        if not sdf_file.exists():
            sdf_candidates = list(pdb_file.parent.glob('*.sdf'))
            sdf_file = sdf_candidates[0] if sdf_candidates else None
            if not sdf_file:
                continue
        pockets.append({
            'name': pdb_file.stem,
            'protein': str(pdb_file),
            'ref_ligand': str(sdf_file),
        })
    if max_pockets:
        pockets = pockets[:max_pockets]
    return pockets


def load_model(checkpoint_path, device, n_steps=500):
    model = DrugFlow.load_from_checkpoint(
        checkpoint_path, map_location=device, strict=False
    )
    model.setup(stage='generation')
    model.batch_size = 1
    model.eval_batch_size = 1
    model.eval().to(device)
    model.T = n_steps
    return model


def process_pocket(protein_path, ligand_path, model, device):
    pdb_model = PDBParser(QUIET=True).get_structure('', protein_path)[0]
    rdmol = Chem.SDMolSupplier(str(ligand_path), sanitize=True)[0]
    if rdmol is None:
        rdmol = Chem.SDMolSupplier(str(ligand_path), sanitize=False)[0]
    ligand, pocket = process_raw_pair(
        pdb_model, rdmol, dist_cutoff=8.0,
        pocket_representation=model.pocket_representation,
        compute_nerf_params=True, nma_input=None
    )
    return {
        'ligand': TensorDict(**ligand),
        'pocket': Residues(**pocket),
    }


def compute_centroid_metrics(rdmols):
    """Compute centroid variance and mean pairwise distance."""
    centroids = []
    for mol in rdmols:
        if mol is None:
            continue
        try:
            conf = mol.GetConformer()
            coords = conf.GetPositions()
            centroids.append(coords.mean(axis=0))
        except Exception:
            continue

    if len(centroids) < 2:
        return 0.0, 0.0

    centroids = np.array(centroids)
    centroid_var = float(np.var(centroids, axis=0).mean())

    # Mean pairwise distance
    dists = []
    for i in range(len(centroids)):
        d = np.linalg.norm(centroids[i] - centroids[i+1:], axis=1)
        dists.extend(d.tolist())
    mean_pair_dist = float(np.mean(dists)) if dists else 0.0

    return centroid_var, mean_pair_dist


# ---------------------------------------------------------------------------
# Experiment 1: No-Annealing Ablation
# ---------------------------------------------------------------------------

def run_no_annealing_ablation(args):
    """Compare t_on=0.5 (default) vs t_on=1.0 (full guidance)."""
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'Testing t_on ∈ {{0.5, 1.0}} on {args.max_pockets} pockets')

    model = load_model(args.checkpoint, device, args.n_steps)

    test_dir = Path(args.test_dir)
    pockets = find_test_pockets(test_dir, args.max_pockets)

    # Pre-process all pockets
    pocket_data = []
    for p in pockets:
        try:
            data = process_pocket(p['protein'], p['ref_ligand'], model, device)
            data = {k: v.to(device) for k, v in data.items()}
            pocket_data.append((p['name'], data))
        except Exception as e:
            print(f'  Skip {p["name"]}: {e}')

    results = {}
    for t_on in [0.5, 1.0]:
        label = f't_on={t_on}'
        print(f'\n--- {label} ---')

        sampler = SVFlowSampler(
            model=model,
            n_trajectories=args.n_trajectories,
            lambda_max=args.lambda_max,
            t_on=t_on,
        )

        variant_results = []
        for pocket_name, data in tqdm(pocket_data, desc=label):
            try:
                rdmols, _, info = sampler.sample(
                    pocket_data=data,
                    timesteps=args.n_steps,
                    save_trajectory_kpe=True,
                )

                n_valid = sum(1 for m in rdmols if m is not None)
                centroid_var, mean_pair_dist = compute_centroid_metrics(rdmols)

                variant_results.append({
                    'pocket': pocket_name,
                    'n_valid': n_valid,
                    'n_total': len(rdmols),
                    'centroid_variance': centroid_var,
                    'mean_pairwise_distance': mean_pair_dist,
                })
            except Exception as e:
                print(f'  Failed {pocket_name}: {e}')

        results[label] = variant_results

    # Summary
    print(f'\n{"="*60}')
    print('No-Annealing Ablation Summary')
    print(f'{"="*60}')
    for label, res in results.items():
        n_valid = [r['n_valid'] for r in res]
        cv = [r['centroid_variance'] for r in res if r['centroid_variance'] > 0]
        print(f'  {label}: validity={np.mean(n_valid):.1f}, '
              f'centroid_var={np.mean(cv):.4f}')

    output_path = Path(args.output_dir) / 'no_annealing_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'Saved to {output_path}')


# ---------------------------------------------------------------------------
# Experiment 2: N-Trajectory Scalability
# ---------------------------------------------------------------------------

def run_scalability(args):
    """Test spatial diversity vs number of trajectories N."""
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    N_values = [2, 4, 8, 16, 32]
    print(f'Testing N ∈ {N_values} on {args.max_pockets} pockets')

    model = load_model(args.checkpoint, device, args.n_steps)

    test_dir = Path(args.test_dir)
    pockets = find_test_pockets(test_dir, args.max_pockets)

    # Pre-process pockets
    pocket_data = []
    for p in pockets:
        try:
            data = process_pocket(p['protein'], p['ref_ligand'], model, device)
            data = {k: v.to(device) for k, v in data.items()}
            pocket_data.append((p['name'], data))
        except Exception as e:
            print(f'  Skip {p["name"]}: {e}')

    results = {}
    for N in N_values:
        label = f'N={N}'
        print(f'\n--- {label} ---')

        sampler = SVFlowSampler(
            model=model,
            n_trajectories=N,
            lambda_max=args.lambda_max,
            t_on=args.t_on,
        )

        t_start = time.time()
        variant_results = []
        for pocket_name, data in tqdm(pocket_data, desc=label):
            try:
                rdmols, _, info = sampler.sample(
                    pocket_data=data,
                    timesteps=args.n_steps,
                )

                n_valid = sum(1 for m in rdmols if m is not None)
                centroid_var, mean_pair_dist = compute_centroid_metrics(rdmols)

                variant_results.append({
                    'pocket': pocket_name,
                    'n_valid': n_valid,
                    'n_total': len(rdmols),
                    'centroid_variance': centroid_var,
                    'mean_pairwise_distance': mean_pair_dist,
                })
            except Exception as e:
                print(f'  Failed {pocket_name}: {e}')

        elapsed = time.time() - t_start
        results[label] = {
            'results': variant_results,
            'time_seconds': elapsed,
            'time_per_pocket': elapsed / len(pocket_data) if pocket_data else 0,
        }

    # Summary
    print(f'\n{"="*60}')
    print('Scalability Summary')
    print(f'{"="*60}')
    for label, data in results.items():
        res = data['results']
        cv = [r['centroid_variance'] for r in res if r['centroid_variance'] > 0]
        md = [r['mean_pairwise_distance'] for r in res if r['mean_pairwise_distance'] > 0]
        print(f'  {label}: centroid_var={np.mean(cv):.4f}, '
              f'pair_dist={np.mean(md):.4f}, '
              f'time={data["time_seconds"]:.1f}s')

    # Marginal diversity gain
    print(f'\n{"="*60}')
    print('Marginal Diversity Gain')
    print(f'{"="*60}')
    prev_cv = None
    for N in N_values:
        label = f'N={N}'
        res = results[label]['results']
        cv = np.mean([r['centroid_variance'] for r in res if r['centroid_variance'] > 0])
        if prev_cv is not None and prev_cv > 0:
            gain = (cv - prev_cv) / prev_cv * 100
            print(f'  {label}: Δ centroid_var = {gain:+.1f}%')
        prev_cv = cv

    output_path = Path(args.output_dir) / 'scalability_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'Saved to {output_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Supplementary Ablation Experiments')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test_dir', type=str,
                        default='/root/autodl-tmp/data/test_sets/CrossDocked_test_set')
    parser.add_argument('--output_dir', type=str, default='./ablation_extras')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['no_annealing', 'scalability', 'all'])
    parser.add_argument('--n_trajectories', type=int, default=10)
    parser.add_argument('--n_steps', type=int, default=500)
    parser.add_argument('--lambda_max', type=float, default=1.0)
    parser.add_argument('--t_on', type=float, default=0.5)
    parser.add_argument('--max_pockets', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    utils.set_deterministic(seed=args.seed)
    utils.disable_rdkit_logging()
    warnings.filterwarnings('ignore')

    if args.mode in ('no_annealing', 'all'):
        run_no_annealing_ablation(args)

    if args.mode in ('scalability', 'all'):
        run_scalability(args)


if __name__ == '__main__':
    main()
