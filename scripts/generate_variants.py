#!/usr/bin/env python3
"""
Batch Ablation Variants Runner

Runs all 5 SV-Flow ablation variants on a subset of pockets for diagnostic
comparison. Used for Appendix A ablation study.

Variants:
  1. SV-Flow Core  — SVGD only (our recommended method)
  2. SV-Flow FULL  — SVGD + Tangent Projection + Orthogonal Preservation
  3. w/o TP         — SVGD + OP only (no tangent projection)
  4. w/o OP         — SVGD + TP only (no orthogonal preservation)
  5. Isotropic      — 1/r² distance repulsion (Metadiffusion-style baseline)

Usage:
    python scripts/generate_variants.py \
        --checkpoint /root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt \
        --max_pockets 10 --n_trajectories 10 --n_steps 500 \
        --output_dir ./ablation_study
"""

import argparse
import sys
import os
import json
import warnings
import time
from pathlib import Path

import torch
import numpy as np

# Fix for PyTorch 2.6+
torch.serialization.add_safe_globals([argparse.Namespace])

# Add paths
basedir = Path(__file__).resolve().parent.parent
drugflow_path = Path('/root/baselines/DrugFlow/code/DrugFlow-main')
sys.path.insert(0, str(drugflow_path))
sys.path.insert(0, str(basedir))

from src import utils
from src.model.lightning import DrugFlow
from src.data.data_utils import process_raw_pair, TensorDict, Residues
from src.analysis.visualization_utils import mols_to_pdbfile
from Bio.PDB import PDBParser
from rdkit import Chem
from tqdm import tqdm

from svflow.sampler import SVFlowSampler


VARIANTS = {
    'core': {
        'label': 'SV-Flow Core (SVGD kernel)',
        'use_svgd_kernel': True,
    },
    'isotropic': {
        'label': 'Isotropic Repulsion (1/r² baseline)',
        'use_svgd_kernel': False,
    },
    'no_annealing': {
        'label': 'w/o Time Annealing (t_on=1.0)',
        'use_svgd_kernel': True,
        't_on': 1.0,
    },
}


def find_test_pockets(test_dir: Path):
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


def load_model(checkpoint_path, device, n_steps):
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
    ligand['name'] = 'ligand'
    return {'ligand': ligand, 'pocket': pocket}


def main():
    parser = argparse.ArgumentParser(description='Ablation Variants Batch Runner')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test_dir', type=str,
                        default='/root/autodl-tmp/data/test_sets/CrossDocked_test_set')
    parser.add_argument('--output_dir', type=str, default='./ablation_study')
    parser.add_argument('--n_trajectories', type=int, default=10)
    parser.add_argument('--n_steps', type=int, default=500)
    parser.add_argument('--lambda_max', type=float, default=1.0)
    parser.add_argument('--t_on', type=float, default=0.5)
    parser.add_argument('--max_pockets', type=int, default=3)
    parser.add_argument('--variants', type=str, default='core,full,no_tp,no_op,isotropic',
                        help='Comma-separated variant keys to run')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    utils.set_deterministic(seed=args.seed)
    utils.disable_rdkit_logging()
    warnings.filterwarnings('ignore')

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Load model once
    model = load_model(args.checkpoint, device, args.n_steps)

    # Find pockets
    pockets = find_test_pockets(Path(args.test_dir))
    pockets = pockets[:args.max_pockets]
    print(f'Running on {len(pockets)} pockets')

    # Process pocket data once
    pocket_data_list = []
    for p in pockets:
        try:
            data = process_pocket(p['protein'], p['ref_ligand'], model, device)
            data = {
                'ligand': TensorDict(**data['ligand']).to(device),
                'pocket': Residues(**data['pocket']).to(device),
            }
            pocket_data_list.append((p['name'], data))
        except Exception as e:
            print(f'Failed to process {p["name"]}: {e}')

    variant_keys = [v.strip() for v in args.variants.split(',')]
    all_results = {}

    for vkey in variant_keys:
        vcfg = VARIANTS[vkey]
        print(f'\n{"="*60}')
        print(f'Running variant: {vcfg["label"]} ({vkey})')
        print(f'{"="*60}')

        t_on = vcfg.get('t_on', args.t_on)
        sampler = SVFlowSampler(
            model=model,
            n_trajectories=args.n_trajectories,
            lambda_max=args.lambda_max,
            t_on=t_on,
            use_svgd_kernel=vcfg['use_svgd_kernel'],
        )

        variant_results = []
        for pocket_name, data in tqdm(pocket_data_list, desc=vcfg['label']):
            try:
                rdmols, rdpockets, info = sampler.sample(
                    pocket_data=data,
                    timesteps=args.n_steps,
                    save_trajectory_kpe=True,
                )

                # Save to variant-specific directory
                outdir = Path(args.output_dir) / vkey / pocket_name
                outdir.mkdir(parents=True, exist_ok=True)
                for i, mol in enumerate(rdmols):
                    if mol is not None:
                        utils.write_sdf_file(str(outdir / f'mol_{i:02d}.sdf'), [mol])
                mols_to_pdbfile(rdpockets, str(outdir / 'pocket.pdb'))

                variant_results.append({
                    'pocket': pocket_name,
                    'n_valid': sum(1 for m in rdmols if m is not None),
                    'sizes': info['ligand_sizes'],
                    'kpe': info.get('kpe_per_trajectory', []),
                })
            except Exception as e:
                print(f'  Failed {pocket_name}: {e}')
                variant_results.append({'pocket': pocket_name, 'error': str(e)})

        all_results[vkey] = variant_results

    # Save summary
    summary_path = Path(args.output_dir) / 'ablation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print quick summary
    print(f'\n{"="*60}')
    print('Ablation Summary')
    print(f'{"="*60}')
    for vkey in variant_keys:
        results = all_results[vkey]
        n_valid = sum(r.get('n_valid', 0) for r in results)
        print(f'  {VARIANTS[vkey]["label"]:35s}: {n_valid} valid molecules')

    print(f'\nResults saved to {summary_path}')


if __name__ == '__main__':
    main()
