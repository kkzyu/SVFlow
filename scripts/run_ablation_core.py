#!/usr/bin/env python3
"""
SV-Flow 核心变体消融实验 — 批量运行脚本

运行 5 个变体 × 5 个口袋，每个变体生成 N=10 个分子。

变体:
  1. core       — 纯 SVGD + 时间退火 (基准, 推荐配置)
  2. full       — SVGD + TP + OP
  3. wo_op      — SVGD + TP (移除正交保护)
  4. wo_tp      — SVGD + OP (移除切平面投影)
  5. isotropic  — 1/r² 各向同性排斥 (Metadiffusion 基线)

输出结构:
  ./output/ablation_core/
    core/        {pocket_name}/mol_00.sdf ... mol_09.sdf + pocket.pdb
    full/        ...
    wo_op/       ...
    wo_tp/       ...
    isotropic/   ...
    ablation_summary.json

用法:
    python scripts/run_ablation_core.py \
        --checkpoint /root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt \
        --output_dir ./output/ablation_core \
        --n_trajectories 10 --n_steps 500 \
        --max_workers 1 --device cuda:0
"""

import argparse
import sys
import json
import time
import warnings
from pathlib import Path

import torch
import numpy as np

# Fix for PyTorch 2.6+ weights_only default
torch.serialization.add_safe_globals([argparse.Namespace, __import__('pathlib').PosixPath])

# Add DrugFlow to path
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
from svflow.variants import get_config, describe_variant


# ── 5 representative CrossDocked test pockets ──────────────────────────
# Selected to cover diverse protein families and pocket geometries.
REPRESENTATIVE_POCKETS = [
    'ABL2_HUMAN_274_551_0',    # Kinase (ABL2) — medium/large pocket
    'ACE_HUMAN_650_1230_0',    # Peptidase (ACE) — irregular pocket
    'AK1BA_HUMAN_1_316_0',     # Aldo-keto reductase — enclosed pocket
    'AKT1_HUMAN_1_137_0',      # Kinase (AKT1) — medium pocket
    'AROE_THET8_1_263_0',      # Shikimate dehydrogenase — small pocket
]


def find_pockets_in_test_dir(test_dir: Path, pocket_dir_names: list) -> list:
    """Locate the PDB + SDF files for specified pocket directories."""
    pockets = []
    for dirname in pocket_dir_names:
        pdir = test_dir / dirname
        if not pdir.is_dir():
            print(f'  ⚠  Directory not found: {pdir}')
            continue
        pdb_files = sorted(pdir.glob('*.pdb'))
        sdf_files = sorted(pdir.glob('*.sdf'))
        if not pdb_files or not sdf_files:
            print(f'  ⚠  Missing PDB or SDF in: {pdir}')
            continue
        pockets.append({
            'name': dirname,
            'protein': str(pdb_files[0]),
            'ref_ligand': str(sdf_files[0]),
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


def process_pocket(protein_path: str, ligand_path: str, model, device: str) -> dict:
    """Process a protein-ligand pair into model input format."""
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
    parser = argparse.ArgumentParser(description='SV-Flow Core Variant Ablation')
    parser.add_argument('--checkpoint', type=str,
                        default='/root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt',
                        help='Path to DrugFlow checkpoint')
    parser.add_argument('--test_dir', type=str,
                        default='/root/autodl-tmp/data/test_sets/CrossDocked_test_set',
                        help='Path to CrossDocked test set')
    parser.add_argument('--output_dir', type=str, default='./output/ablation_core',
                        help='Output directory')
    parser.add_argument('--pockets', type=str, default=None,
                        help='Comma-separated pocket dir names (uses defaults if omitted)')
    parser.add_argument('--variants', type=str,
                        default='core,full,wo_op,wo_tp,isotropic',
                        help='Comma-separated variant keys')
    parser.add_argument('--n_trajectories', type=int, default=10,
                        help='Number of coupled trajectories (N)')
    parser.add_argument('--n_steps', type=int, default=500,
                        help='ODE integration steps (T)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Setup
    utils.set_deterministic(seed=args.seed)
    utils.disable_rdkit_logging()
    warnings.filterwarnings('ignore')

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'Variants: {args.variants}')
    print(f'N={args.n_trajectories}, T={args.n_steps}')

    # Load model once
    model = load_model(args.checkpoint, device, args.n_steps)

    # Resolve pocket list
    test_dir = Path(args.test_dir)
    pocket_names = (args.pockets.split(',') if args.pockets
                    else REPRESENTATIVE_POCKETS)
    pockets = find_pockets_in_test_dir(test_dir, pocket_names)
    print(f'Pockets: {len(pockets)}')

    # Pre-process all pockets (reused across variants)
    pocket_data_list = []
    for p in pockets:
        try:
            data = process_pocket(p['protein'], p['ref_ligand'], model, device)
            data = {
                'ligand': TensorDict(**data['ligand']).to(device),
                'pocket': Residues(**data['pocket']).to(device),
            }
            pocket_data_list.append((p['name'], data))
            print(f'  ✓ {p["name"]}')
        except Exception as e:
            print(f'  ✗ {p["name"]}: {e}')

    if not pocket_data_list:
        print('ERROR: No pockets loaded. Aborting.')
        sys.exit(1)

    # Output root
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_keys = [v.strip() for v in args.variants.split(',')]
    all_results = {}

    for vkey in variant_keys:
        cfg = get_config(vkey)
        label = describe_variant(vkey)
        print(f'\n{"="*60}')
        print(f'  Variant: {label}')
        print(f'  Config: {cfg}')
        print(f'{"="*60}')

        # Build sampler for this variant
        sampler = SVFlowSampler(
            model=model,
            n_trajectories=args.n_trajectories,
            **cfg,
            verbose=False,
        )

        variant_results = []
        t_start = time.time()

        for pocket_name, data in tqdm(pocket_data_list, desc=f'{vkey}'):
            try:
                rdmols, rdpockets, info = sampler.sample(
                    pocket_data=data,
                    timesteps=args.n_steps,
                    save_trajectory_kpe=True,
                )

                # Save to output
                outdir = output_dir / vkey / pocket_name
                outdir.mkdir(parents=True, exist_ok=True)

                for i, mol in enumerate(rdmols):
                    if mol is not None:
                        utils.write_sdf_file(str(outdir / f'mol_{i:02d}.sdf'), [mol])

                # Save pocket PDB for evaluation (clash detection)
                mols_to_pdbfile(rdpockets, str(outdir / 'pocket.pdb'))

                n_valid = sum(1 for m in rdmols if m is not None)
                variant_results.append({
                    'pocket': pocket_name,
                    'n_total': len(rdmols),
                    'n_valid': n_valid,
                    'ligand_sizes': info['ligand_sizes'],
                    'kpe': info.get('kpe_per_trajectory', []),
                    'status': 'success',
                })
            except Exception as e:
                print(f'\n  Failed {vkey}/{pocket_name}: {e}')
                import traceback
                traceback.print_exc()
                variant_results.append({
                    'pocket': pocket_name,
                    'status': 'failed',
                    'error': str(e),
                })

        elapsed = time.time() - t_start
        n_success = sum(1 for r in variant_results if r.get('status') == 'success')
        n_mols = sum(r.get('n_valid', 0) for r in variant_results)
        print(f'  Done in {elapsed:.0f}s — {n_mols} valid mols '
              f'across {n_success}/{len(variant_results)} pockets')

        all_results[vkey] = {
            'label': label,
            'config': cfg,
            'results': variant_results,
            'elapsed_seconds': elapsed,
            'n_successful_pockets': n_success,
            'n_total_molecules': n_mols,
        }

    # ── Save aggregate summary ──────────────────────────────────────────
    summary_path = output_dir / 'ablation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\n✅ Ablation summary saved to {summary_path}')

    # ── Quick summary table ─────────────────────────────────────────────
    print(f'\n{"="*70}')
    print('ABLATION COMPLETE — Summary')
    print(f'{"="*70}')
    print(f'{"Variant":20s} {"Pockets":>8s} {"Mols":>8s} {"Time":>8s}')
    print('-' * 48)
    for vkey in variant_keys:
        info = all_results[vkey]
        print(f'{vkey:20s} {info["n_successful_pockets"]:>8d} '
              f'{info["n_total_molecules"]:>8d} {info["elapsed_seconds"]:>7.0f}s')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
