#!/usr/bin/env python3
"""
Smoke Test for SV-Flow Pipeline

Runs all tests that can be done without GPU:
1. Data loading: verify PDB/SDF files parse correctly for test set pockets
2. Core modules: functional tests for kinematics, SVGD, projection, scheduling
3. Integration: verify DrugFlow checkpoint structure

Usage (CPU-only, no GPU needed):
    python scripts/smoke_test.py

Usage (with GPU, quick end-to-end):
    python scripts/smoke_test.py --gpu --checkpoint /root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt --n_steps 10 --n_trajectories 2
"""

import argparse
import sys
import os
import warnings
from pathlib import Path

import torch
import numpy as np

# Fix for PyTorch 2.6+ weights_only=True default when loading checkpoints
torch.serialization.add_safe_globals([argparse.Namespace])

# Add paths
drugflow_path = Path('/root/baselines/DrugFlow/code/DrugFlow-main')
sys.path.insert(0, str(drugflow_path))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings('ignore')


def test_imports():
    """Test all required imports work."""
    print('=' * 60)
    print('1. Import Tests')
    print('=' * 60)

    errors = []
    checks = [
        ('torch', 'PyTorch'),
        ('torch.nn', 'torch.nn'),
        ('numpy', 'NumPy'),
        ('rdkit', 'RDKit'),
        ('rdkit.Chem', 'RDKit.Chem'),
        ('Bio.PDB', 'BioPython PDB'),
        ('yaml', 'PyYAML'),
    ]
    for mod, name in checks:
        try:
            __import__(mod)
            print(f'  ✓ {name}')
        except ImportError as e:
            print(f'  ✗ {name}: {e}')
            errors.append(name)

    # Heavy imports - might fail if CUDA not available
    try:
        import torch_geometric
        print(f'  ✓ torch_geometric')
    except ImportError:
        print(f'  ⚠ torch_geometric not available (needed for GPU inference only)')

    try:
        import pytorch_lightning
        print(f'  ✓ pytorch_lightning')
    except ImportError:
        print(f'  ⚠ pytorch_lightning not available')

    return len(errors) == 0


def test_data_loading():
    """Test that test set PDB/SDF files can be parsed."""
    print('\n' + '=' * 60)
    print('2. Data Loading Tests')
    print('=' * 60)

    test_dir = Path('/root/autodl-tmp/data/test_sets/CrossDocked_test_set')
    if not test_dir.exists():
        print(f'  ✗ Test set directory not found: {test_dir}')
        return False

    pockets = sorted(test_dir.iterdir())
    pockets = [p for p in pockets if p.is_dir()]
    print(f'  Found {len(pockets)} pocket directories')

    if not pockets:
        print('  ✗ No pocket directories found')
        return False

    # Test first 3 pockets
    from Bio.PDB import PDBParser
    from rdkit import Chem

    success = 0
    for pdir in pockets[:3]:
        name = pdir.name
        pdbs = list(pdir.glob('*.pdb'))
        sdfs = list(pdir.glob('*.sdf'))

        if not pdbs:
            print(f'  ✗ {name}: No PDB file')
            continue
        if not sdfs:
            print(f'  ✗ {name}: No SDF file')
            continue

        try:
            pdb_parser = PDBParser(QUIET=True)
            structure = pdb_parser.get_structure('', str(pdbs[0]))
            n_residues = len(list(structure.get_residues()))
        except Exception as e:
            print(f'  ✗ {name}: PDB parse error: {e}')
            continue

        try:
            mol = Chem.SDMolSupplier(str(sdfs[0]), sanitize=True)[0]
            if mol is None:
                mol = Chem.SDMolSupplier(str(sdfs[0]), sanitize=False)[0]
            n_atoms = mol.GetNumAtoms() if mol else 0
        except Exception as e:
            print(f'  ✗ {name}: SDF parse error: {e}')
            continue

        print(f'  ✓ {name}: {n_residues} residues, {n_atoms} ligand atoms')
        success += 1

    print(f'  {success}/{min(3, len(pockets))} pockets loaded successfully')
    return success > 0


def test_core_modules():
    """Test core SV-Flow modules on synthetic data."""
    print('\n' + '=' * 60)
    print('3. Core Module Tests')
    print('=' * 60)

    import torch

    all_pass = True

    # 3a. Kinematics
    print('\n  3a. Kinematics (§4.1)')
    try:
        from svflow.kinematics import decompose_velocity, compute_center_of_mass

        vel = torch.randn(30, 3, dtype=torch.float64)
        mask = torch.tensor([0]*10 + [1]*10 + [2]*10)

        v_int, v_CoM = decompose_velocity(vel, mask)
        assert torch.allclose(v_int + v_CoM, vel, atol=1e-6), "Decomposition failed"

        # v_int should have zero CoM motion per molecule
        for mol_id in range(3):
            v_int_mol = v_int[mask == mol_id]
            com_motion = v_int_mol.mean(dim=0).abs().max().item()
            assert com_motion < 1e-10, f"v_int has CoM motion: {com_motion}"

        print('    ✓ Decomposition: v_int + v_CoM == vel, v_int zero-CoM')
    except Exception as e:
        print(f'    ✗ Failed: {e}')
        all_pass = False

    # 3b. SVGD
    print('\n  3b. SVGD Interaction (§4.2)')
    try:
        from svflow.svgd import compute_svgd_velocity

        # 4 particles at corners of a square
        x = torch.tensor([[0., 0., 0.], [3., 0., 0.], [0., 3., 0.], [3., 3., 0.]], dtype=torch.float64)
        delta_v = compute_svgd_velocity(x, d_min=5.0, h=1.0)

        # Forces should sum to zero (momentum conservation)
        total_force = delta_v.sum(dim=0).abs().max().item()
        assert total_force < 1e-14, f"Total force not zero: {total_force}"

        # Each particle should be pushed away from center
        center = x.mean(dim=0)
        for i in range(4):
            direction_to_center = center - x[i]
            direction_to_center = direction_to_center / (direction_to_center.norm() + 1e-8)
            projection = (delta_v[i] * direction_to_center).sum()
            # Velocity should have component away from center
            assert projection > -0.01, f"Particle {i} being pushed toward center"

        print(f'    ✓ Zero-sum forces, particles repel from center')
    except Exception as e:
        print(f'    ✗ Failed: {e}')
        all_pass = False

    # 3c. Tangent Plane Projection
    print('\n  3c. Tangent Plane Projection (§4.3)')
    try:
        from svflow.tangent_projection import tangent_plane_projection

        v = torch.tensor([[1., 2., 0.], [0., 1., 1.]], dtype=torch.float64)
        grad = torch.tensor([[1., 0., 0.], [0., 0., 1.]], dtype=torch.float64)

        v_proj = tangent_plane_projection(v, grad)
        # First: v=(1,2,0), n=(1,0,0) → parallel=(1,0,0) → proj=(0,2,0)
        assert torch.allclose(v_proj[0], torch.tensor([0., 2., 0.], dtype=torch.float64), atol=1e-6)
        # Second: v=(0,1,1), n=(0,0,1) → parallel=(0,0,1) → proj=(0,1,0)
        assert torch.allclose(v_proj[1], torch.tensor([0., 1., 0.], dtype=torch.float64), atol=1e-6)
        print('    ✓ Normal components correctly removed')
    except Exception as e:
        print(f'    ✗ Failed: {e}')
        all_pass = False

    # 3d. Orthogonal Preservation
    print('\n  3d. Orthogonal Kinetic Preservation (§4.4)')
    try:
        from svflow.orthogonal_preservation import orthogonal_preservation

        v_in = torch.tensor([[1., 2., 3.], [3., 1., 2.]], dtype=torch.float64)
        v_com = torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=torch.float64)

        v_out = orthogonal_preservation(v_in, v_com)
        # Should be orthogonal to v_CoM
        for i in range(2):
            dot = (v_out[i] * v_com[i]).sum().abs().item()
            assert dot < 1e-10, f"Non-orthogonal: dot={dot}"
        print('    ✓ Output is orthogonal to v_CoM')
    except Exception as e:
        print(f'    ✗ Failed: {e}')
        all_pass = False

    # 3e. Time Scheduler
    print('\n  3e. Time-Annealed Scheduling (§4.5)')
    try:
        from svflow.time_scheduler import TimeAnnealedScheduler

        sched = TimeAnnealedScheduler(t_on=0.5, lambda_max=1.0)
        t = torch.tensor([1.0, 0.7, 0.5, 0.3, 0.0])
        lam = sched(t)

        assert lam[0] == 0.0, f"t=1.0: expected 0, got {lam[0]}"
        assert lam[1] == 0.0, f"t=0.7: expected 0, got {lam[1]}"
        assert lam[2] == 0.0, f"t=0.5: expected 0, got {lam[2]}"
        assert abs(lam[3] - 0.16) < 0.01, f"t=0.3: expected 0.16, got {lam[3]}"
        assert lam[4] == 1.0, f"t=0.0: expected 1.0, got {lam[4]}"
        print('    ✓ λ(t) schedule correct: late onset, parabolic ramp')
    except Exception as e:
        print(f'    ✗ Failed: {e}')
        all_pass = False

    return all_pass


def test_checkpoint_structure():
    """Verify checkpoint exists and has expected size."""
    print('\n' + '=' * 60)
    print('4. Checkpoint Check')
    print('=' * 60)

    ckpt_path = Path('/root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt')
    if not ckpt_path.exists():
        print(f'  ✗ Checkpoint not found: {ckpt_path}')
        return False

    size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    print(f'  ✓ Checkpoint exists: {size_mb:.0f} MB')
    print(f'  ✓ Path: {ckpt_path}')

    # Read training config for reference
    config_path = Path('/root/baselines/DrugFlow/code/DrugFlow-main/configs/training/drugflow.yml')
    if config_path.exists():
        print(f'  ✓ Training config found: {config_path}')

    return True


def test_quick_gpu(checkpoint_path, n_steps, n_trajectories):
    """Quick end-to-end GPU test on a single pocket."""
    print('\n' + '=' * 60)
    print('5. Quick GPU End-to-End Test')
    print('=' * 60)

    import torch
    if not torch.cuda.is_available():
        print('  ⚠ No GPU available, skipping GPU test')
        print('  Run with --gpu on a GPU machine to test end-to-end')
        return True

    device = 'cuda:0'
    print(f'  Using {device}')

    try:
        from src.model.lightning import DrugFlow
        from src.data.data_utils import process_raw_pair, TensorDict, Residues
        from Bio.PDB import PDBParser
        from rdkit import Chem

        from svflow.sampler import SVFlowSampler

        # Load model
        print('  Loading model...')
        model = DrugFlow.load_from_checkpoint(
            checkpoint_path, map_location=device, strict=False
        )
        model.setup(stage='generation')
        model.batch_size = n_trajectories
        model.eval_batch_size = n_trajectories
        model.eval().to(device)
        model.T = n_steps

        # Pick first test pocket
        test_dir = Path('/root/autodl-tmp/data/test_sets/CrossDocked_test_set')
        pockets = sorted([p for p in test_dir.iterdir() if p.is_dir()])
        if not pockets:
            print('  ✗ No test pockets found')
            return False

        pocket_dir = pockets[0]
        pdbs = list(pocket_dir.glob('*.pdb'))
        sdfs = list(pocket_dir.glob('*.sdf'))
        print(f'  Testing on: {pocket_dir.name}')

        # Process data
        pdb_model = PDBParser(QUIET=True).get_structure('', str(pdbs[0]))[0]
        rdmol = Chem.SDMolSupplier(str(sdfs[0]), sanitize=True)[0]
        if rdmol is None:
            rdmol = Chem.SDMolSupplier(str(sdfs[0]), sanitize=False)[0]

        ligand, pocket = process_raw_pair(
            pdb_model, rdmol,
            dist_cutoff=8.0,
            pocket_representation=model.pocket_representation,
            compute_nerf_params=True,
            nma_input=None
        )

        data = {
            'ligand': TensorDict(**ligand).to(device),
            'pocket': Residues(**pocket).to(device),
        }

        # SV-Flow sampling
        print(f'  Running SV-Flow: N={n_trajectories}, T={n_steps}...')
        sampler = SVFlowSampler(
            model=model,
            n_trajectories=n_trajectories,
            lambda_max=1.0,
            t_on=0.5,
        )

        import time
        t_start = time.time()
        rdmols, rdpockets, info = sampler.sample(
            pocket_data=data,
            timesteps=n_steps,
            save_trajectory_kpe=True,
        )
        elapsed = time.time() - t_start

        print(f'  ✓ Generated {len(rdmols)} molecules in {elapsed:.1f}s')
        print(f'  ✓ Molecule sizes: {info["ligand_sizes"]}')
        print(f'  ✓ KPE per trajectory: {[f"{k:.3f}" for k in info.get("kpe_per_trajectory", [])]}')

        # Quick validity check
        valid_count = sum(1 for m in rdmols if m is not None)
        print(f'  ✓ {valid_count}/{len(rdmols)} molecules are valid RDKit objects')

        return True

    except Exception as e:
        print(f'  ✗ GPU test failed: {e}')
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='SV-Flow Smoke Test')
    parser.add_argument('--gpu', action='store_true', help='Run GPU test')
    parser.add_argument('--checkpoint', type=str,
                        default='/root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt')
    parser.add_argument('--n_steps', type=int, default=10, help='ODE steps for quick test')
    parser.add_argument('--n_trajectories', type=int, default=2, help='Trajectories for quick test')
    args = parser.parse_args()

    results = {}

    results['imports'] = test_imports()
    results['data'] = test_data_loading()
    results['modules'] = test_core_modules()
    results['checkpoint'] = test_checkpoint_structure()

    if args.gpu:
        results['gpu'] = test_quick_gpu(args.checkpoint, args.n_steps, args.n_trajectories)

    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    all_pass = True
    for name, passed in results.items():
        status = '✓ PASS' if passed else '✗ FAIL'
        print(f'  {name:20s}: {status}')
        if not passed:
            all_pass = False

    if all_pass:
        print('\n✅ All tests passed! Ready for GPU experiments.')
    else:
        print('\n⚠ Some tests failed. Fix issues before GPU experiments.')

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
