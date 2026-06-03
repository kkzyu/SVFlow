#!/usr/bin/env python3
"""
DrugFlow Baseline Generation Script

Generates molecules using DrugFlow's native independent sampling (no SVGD coupling)
for head-to-head comparison with SV-Flow.

Usage:
    # Standard baseline (N=10, T=500)
    python scripts/generate_baseline.py \
        --checkpoint /root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt \
        --output_dir ./drugflow_baseline_samples \
        --n_samples 10 --n_steps 500

    # N=50 for MaxMin post-processing baseline
    python scripts/generate_baseline.py \
        --checkpoint /root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt \
        --n_samples 50 --n_steps 500 \
        --output_dir ./drugflow_n50_samples
"""

import argparse
import sys
import os
import warnings
from pathlib import Path

import torch
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from Bio.PDB import PDBParser

# Fix for PyTorch 2.6+ weights_only=True default
torch.serialization.add_safe_globals([argparse.Namespace])

# Add DrugFlow to path
basedir = Path(__file__).resolve().parent.parent
drugflow_path = Path('/root/baselines/DrugFlow/code/DrugFlow-main')
sys.path.insert(0, str(drugflow_path))
sys.path.insert(0, str(basedir))

from src import utils
from src.data.data_utils import process_raw_pair, TensorDict, Residues
from src.model.lightning import DrugFlow
from src.analysis.visualization_utils import mols_to_pdbfile


def find_test_pockets(test_dir: Path):
    """Find all pocket/ligand pairs in the test set directory."""
    pockets = []
    for pdb_file in sorted(test_dir.rglob('*.pdb')):
        sdf_file = pdb_file.with_suffix('.sdf')
        if not sdf_file.exists():
            sdf_candidates = list(pdb_file.parent.glob('*.sdf'))
            if sdf_candidates:
                sdf_file = sdf_candidates[0]
            else:
                continue
        pockets.append({
            'name': pdb_file.stem,
            'protein': str(pdb_file),
            'ref_ligand': str(sdf_file),
        })
    return pockets


def load_model(checkpoint_path: str, device: str, n_steps: int = 500):
    """Load pretrained DrugFlow model."""
    print(f'Loading checkpoint from {checkpoint_path}...')
    model = DrugFlow.load_from_checkpoint(
        checkpoint_path, map_location=device, strict=False
    )
    model.setup(stage='generation')
    model.batch_size = 1
    model.eval_batch_size = 1
    model.eval().to(device)
    model.T = n_steps
    return model


def process_pocket_and_ligand(protein_path: str, ligand_path: str, model, device: str):
    """Process a single protein-ligand pair into model input format."""
    pdb_model = PDBParser(QUIET=True).get_structure('', protein_path)[0]
    rdmol = Chem.SDMolSupplier(str(ligand_path), sanitize=True)[0]
    if rdmol is None:
        rdmol = Chem.SDMolSupplier(str(ligand_path), sanitize=False)[0]

    ligand, pocket = process_raw_pair(
        pdb_model, rdmol,
        dist_cutoff=8.0,
        pocket_representation=model.pocket_representation,
        compute_nerf_params=True,
        nma_input=None
    )
    ligand['name'] = 'ligand'
    return {'ligand': ligand, 'pocket': pocket}


def main():
    parser = argparse.ArgumentParser(description='DrugFlow Baseline Generation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to DrugFlow checkpoint')
    parser.add_argument('--test_dir', type=str,
                        default='/root/autodl-tmp/data/test_sets/CrossDocked_test_set',
                        help='Path to CrossDocked test set')
    parser.add_argument('--output_dir', type=str, default='./drugflow_baseline_samples',
                        help='Output directory')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of independent samples per pocket')
    parser.add_argument('--n_steps', type=int, default=500,
                        help='Number of ODE integration steps')
    parser.add_argument('--molecule_size', type=str, default=None,
                        help='Molecule size specification')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pocket_ids', type=str, default=None,
                        help='Comma-separated list of specific pocket IDs')
    parser.add_argument('--max_pockets', type=int, default=None,
                        help='Maximum number of pockets to process')
    args = parser.parse_args()

    utils.set_deterministic(seed=args.seed)
    utils.disable_rdkit_logging()
    warnings.filterwarnings('ignore')

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Load model
    model = load_model(args.checkpoint, device, args.n_steps)

    # Find test pockets
    test_dir = Path(args.test_dir)
    pockets = find_test_pockets(test_dir)
    print(f'Found {len(pockets)} test pockets')

    if args.pocket_ids:
        ids = set(args.pocket_ids.split(','))
        pockets = [p for p in pockets if p['name'] in ids]
        print(f'Filtered to {len(pockets)} pockets')
    if args.max_pockets:
        pockets = pockets[:args.max_pockets]

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_summary = []
    for pocket_info in tqdm(pockets, desc='Generating baseline'):
        pocket_name = pocket_info['name']
        pocket_outdir = output_dir / pocket_name
        pocket_outdir.mkdir(parents=True, exist_ok=True)

        try:
            # Process pocket data
            data = process_pocket_and_ligand(
                pocket_info['protein'], pocket_info['ref_ligand'],
                model, device
            )

            # Move to device
            new_data = {
                'ligand': TensorDict(**data['ligand']).to(device),
                'pocket': Residues(**data['pocket']).to(device),
            }

            # DrugFlow native independent sampling (no SVGD)
            rdmols, rdpockets, _ = model.sample(
                data=new_data,
                n_samples=args.n_samples,
                num_nodes=args.molecule_size,
                timesteps=args.n_steps,
            )

            # Save molecules
            valid_count = 0
            for i, mol in enumerate(rdmols):
                if mol is not None:
                    out_sdf = pocket_outdir / f'mol_{i:02d}.sdf'
                    utils.write_sdf_file(str(out_sdf), [mol])
                    valid_count += 1

            # Save pocket
            out_pocket = pocket_outdir / 'pocket.pdb'
            mols_to_pdbfile(rdpockets, str(out_pocket))

            results_summary.append({
                'pocket': pocket_name,
                'n_generated': len(rdmols),
                'n_successful': valid_count,
            })

        except Exception as e:
            print(f'\nFailed to generate for {pocket_name}: {e}')
            import traceback
            traceback.print_exc()
            results_summary.append({
                'pocket': pocket_name,
                'n_successful': 0,
                'error': str(e),
            })
            continue

    # Save summary
    import json
    summary_path = output_dir / 'generation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)

    n_success = sum(1 for r in results_summary if r.get('n_successful', 0) > 0)
    total_mols = sum(r.get('n_successful', 0) for r in results_summary)
    print(f'\nDone! Generated {total_mols} molecules across {n_success}/{len(pockets)} pockets')
    print(f'Output saved to {output_dir}')


if __name__ == '__main__':
    main()
