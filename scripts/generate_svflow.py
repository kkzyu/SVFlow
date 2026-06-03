#!/usr/bin/env python3
"""
SV-Flow Generation Script

Generates diverse molecules for CrossDocked test set pockets using
Stein Variational Flow Matching (SV-Flow) with multi-trajectory SVGD guidance.

Usage:
    python scripts/generate_svflow.py \
        --checkpoint /root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt \
        --test_dir /root/autodl-tmp/data/test_sets/CrossDocked_test_set \
        --output_dir ./svflow_samples \
        --n_trajectories 10 \
        --n_steps 500 \
        --lambda_max 1.0 \
        --device cuda:0
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

# Fix for PyTorch 2.6+ weights_only=True default when loading checkpoints
# that contain argparse.Namespace and pathlib objects.
torch.serialization.add_safe_globals([
    argparse.Namespace, Path, type(Path()), type(Path().resolve()),
])

# Add DrugFlow to path
basedir = Path(__file__).resolve().parent.parent
drugflow_path = Path('/root/baselines/DrugFlow/code/DrugFlow-main')
sys.path.insert(0, str(drugflow_path))

# Add SVFlow to path
sys.path.insert(0, str(basedir))

from src import utils
from src.data.dataset import ProcessedLigandPocketDataset
from src.data.data_utils import process_raw_pair, TensorDict, Residues
from src.model.lightning import DrugFlow
from src.analysis.visualization_utils import mols_to_pdbfile
from torch.utils.data import DataLoader
from functools import partial

from svflow.sampler import SVFlowSampler


def find_test_pockets(test_dir: Path):
    """Find all pocket/ligand pairs in the test set directory."""
    pockets = []
    for pdb_file in sorted(test_dir.rglob('*.pdb')):
        # Look for corresponding SDF file
        sdf_file = pdb_file.with_suffix('.sdf')
        if not sdf_file.exists():
            # Try to find a ligand file
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
        # Try with sanitize=False
        rdmol = Chem.SDMolSupplier(str(ligand_path), sanitize=False)[0]

    ligand, pocket = process_raw_pair(
        pdb_model, rdmol,
        dist_cutoff=8.0,
        pocket_representation=model.pocket_representation,
        compute_nerf_params=True,
        nma_input=None  # rigid receptor
    )
    ligand['name'] = 'ligand'
    return {'ligand': ligand, 'pocket': pocket}


def main():
    parser = argparse.ArgumentParser(description='SV-Flow Molecular Generation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to DrugFlow checkpoint')
    parser.add_argument('--test_dir', type=str,
                        default='/root/autodl-tmp/data/test_sets/CrossDocked_test_set',
                        help='Path to CrossDocked test set directory')
    parser.add_argument('--output_dir', type=str, default='./svflow_samples',
                        help='Output directory for generated molecules')
    parser.add_argument('--n_trajectories', type=int, default=10,
                        help='Number of coupled SV-Flow trajectories (N)')
    parser.add_argument('--n_steps', type=int, default=500,
                        help='Number of ODE integration steps')
    parser.add_argument('--lambda_max', type=float, default=1.0,
                        help='Maximum SVGD guidance strength')
    parser.add_argument('--t_on', type=float, default=0.5,
                        help='Late-onset time threshold')
    parser.add_argument('--d_min', type=float, default=2.0,
                        help='Distance threshold for repulsive energy (Angstrom)')
    parser.add_argument('--molecule_size', type=str, default=None,
                        help='Molecule size: int, "uniform_low_high", or None (sampled)')
    parser.add_argument('--variant', type=str, default='core',
                        choices=['core', 'isotropic'],
                        help='Method variant: core (SVGD kernel, default), '
                             'isotropic (1/r^2 distance repulsion baseline)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pocket_ids', type=str, default=None,
                        help='Comma-separated list of specific pocket IDs to process')
    parser.add_argument('--max_pockets', type=int, default=None,
                        help='Maximum number of pockets to process')
    args = parser.parse_args()

    # Map variant names to sampler configuration
    VARIANT_CONFIGS = {
        'core':      {'use_svgd': True},
        'isotropic': {'use_svgd': False},
    }
    variant_cfg = VARIANT_CONFIGS[args.variant]

    utils.set_deterministic(seed=args.seed)
    utils.disable_rdkit_logging()
    warnings.filterwarnings('ignore')

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Load model
    model = load_model(args.checkpoint, device, args.n_steps)

    # Create SV-Flow sampler
    sampler = SVFlowSampler(
        model=model,
        n_trajectories=args.n_trajectories,
        lambda_max=args.lambda_max,
        t_on=args.t_on,
        d_min=args.d_min,
        use_svgd_kernel=variant_cfg['use_svgd'],
        verbose=False,
    )
    print(f'Variant: {args.variant} → SVGD kernel={variant_cfg["use_svgd"]}')

    # Find test pockets
    test_dir = Path(args.test_dir)
    pockets = find_test_pockets(test_dir)
    print(f'Found {len(pockets)} test pockets')

    # Filter if needed
    if args.pocket_ids:
        ids = set(args.pocket_ids.split(','))
        pockets = [p for p in pockets if p['name'] in ids]
        print(f'Filtered to {len(pockets)} pockets')
    if args.max_pockets:
        pockets = pockets[:args.max_pockets]

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each pocket
    results_summary = []
    for pocket_info in tqdm(pockets, desc='Generating'):
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

            # SV-Flow sampling
            rdmols, rdpockets, info = sampler.sample(
                pocket_data=new_data,
                num_nodes=args.molecule_size,
                timesteps=args.n_steps,
                save_trajectory_kpe=True,
            )

            # Save molecules
            for i, mol in enumerate(rdmols):
                out_sdf = pocket_outdir / f'mol_{i:02d}.sdf'
                utils.write_sdf_file(str(out_sdf), [mol])

            # Save pocket
            out_pocket = pocket_outdir / 'pocket.pdb'
            mols_to_pdbfile(rdpockets, str(out_pocket))

            # Save info
            results_summary.append({
                'pocket': pocket_name,
                'n_successful': len(rdmols),
                'ligand_sizes': info['ligand_sizes'],
                'kpe': info.get('kpe_per_trajectory', []),
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
