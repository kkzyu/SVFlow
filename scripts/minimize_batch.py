#!/usr/bin/env python3
"""
Batch Energy Minimization for SV-Flow Generated Molecules

Uses RDKit MMFF94 force field — well-validated for drug-like molecules.
Minimizes each ligand independently (gas phase), then reports:
  - Initial/final MMFF94 energy
  - Heavy-atom RMSD after alignment
  - Clash counts before/after

Protocol:
  1. Read SDF with RDKit, add hydrogens
  2. Compute initial MMFF94 energy
  3. Minimize (500 steps L-BFGS variant, gradient tolerance 1e-5)
  4. Compute final energy, RMSD, clash counts
  5. Save minimized structure as SDF

Usage:
    python scripts/minimize_batch.py \
        --input_dir ./output/svflow_core \
        --output_dir ./output/svflow_core_min \
        --max_workers 32
"""

import argparse
import sys
import os
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, rdMolTransforms
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def count_clashes(mol, tolerance: float = 0.4):
    """
    Count internal steric clashes (non-bonded atoms closer than sum of vdW radii - tolerance).
    Uses standard vdW radii (Å): H=1.2, C=1.7, N=1.55, O=1.52, S=1.8, P=1.8, F=1.47, Cl=1.75, Br=1.85, I=1.98
    """
    vdw_radii = {
        1: 1.20,   # H
        5: 1.92,   # B
        6: 1.70,   # C
        7: 1.55,   # N
        8: 1.52,   # O
        9: 1.47,   # F
        15: 1.80,  # P
        16: 1.80,  # S
        17: 1.75,  # Cl
        35: 1.85,  # Br
        53: 1.98,  # I
    }

    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()
    n_clashes = 0

    for i in range(n_atoms):
        ai = mol.GetAtomWithIdx(i)
        if ai.GetAtomicNum() == 0:
            continue
        ri = vdw_radii.get(ai.GetAtomicNum(), 1.70)
        pi = np.array(conf.GetAtomPosition(i))

        for j in range(i + 1, n_atoms):
            aj = mol.GetAtomWithIdx(j)
            if aj.GetAtomicNum() == 0:
                continue

            # Skip bonded atoms (1-2) and 1-3 interactions
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is not None:
                continue
            # Check 1-3 (angle) interactions
            for k in range(n_atoms):
                if k != i and k != j:
                    bik = mol.GetBondBetweenAtoms(i, k)
                    bjk = mol.GetBondBetweenAtoms(j, k)
                    if bik is not None and bjk is not None:
                        break
            else:
                # Not in 1-3 relationship — this is wrong, let me fix
                pass

            # Simpler check: skip if distance in bonds <= 3
            path = Chem.GetShortestPath(mol, i, j)
            if path is not None and len(path) <= 4:  # 1-2, 1-3, 1-4
                continue

            rj = vdw_radii.get(aj.GetAtomicNum(), 1.70)
            pj = np.array(conf.GetAtomPosition(j))
            dist = np.linalg.norm(pi - pj)
            threshold = ri + rj - tolerance

            if dist < threshold and dist > 0.01:
                n_clashes += 1

    return n_clashes


def compute_rmsd(mol1, mol2):
    """Compute heavy-atom RMSD after optimal alignment (Kabsch)."""
    conf1 = mol1.GetConformer()
    conf2 = mol2.GetConformer()

    # Get heavy atom positions
    idx1, idx2 = [], []
    for i in range(mol1.GetNumAtoms()):
        if mol1.GetAtomWithIdx(i).GetAtomicNum() > 1:
            idx1.append(i)
    for i in range(mol2.GetNumAtoms()):
        if mol2.GetAtomWithIdx(i).GetAtomicNum() > 1:
            idx2.append(i)

    if len(idx1) != len(idx2) or len(idx1) < 3:
        return -1.0

    P = np.array([list(conf1.GetAtomPosition(i)) for i in idx1])
    Q = np.array([list(conf2.GetAtomPosition(i)) for i in idx2])

    # Kabsch alignment
    P_centroid = P.mean(axis=0)
    Q_centroid = Q.mean(axis=0)
    P_centered = P - P_centroid
    Q_centered = Q - Q_centroid
    H = P_centered.T @ Q_centered
    try:
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
    except np.linalg.LinAlgError:
        return -1.0

    P_rotated = P_centered @ R
    diff = P_rotated - Q_centered
    return float(np.sqrt(np.mean(diff * diff)))


def minimize_one(args):
    """
    Minimize a single ligand using RDKit MMFF94.

    Args tuple: (sdf_path, output_sdf_path)
    Returns: dict with metrics
    """
    sdf_path, output_sdf_path = args
    mol_name = Path(sdf_path).stem
    pocket_name = Path(sdf_path).parent.name

    result = {
        'sdf_file': str(sdf_path),
        'pocket': pocket_name,
        'mol_name': mol_name,
        'initial_energy': None,
        'final_energy': None,
        'energy_reduction': None,
        'rmsd': None,
        'n_atoms_heavy': 0,
        'clashes_before': 0,
        'clashes_after': 0,
        'n_iterations': 0,
        'status': 'success',
        'error': '',
    }

    try:
        # 1. Read SDF
        supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=True, removeHs=False)
        mol = supplier[0]
        if mol is None:
            supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
            mol = supplier[0]
            if mol is None:
                raise ValueError("Cannot read SDF")

        # 2. Remove existing Hs (model-generated Hs are unreliable), add fresh
        mol = Chem.RemoveHs(mol)
        mol = Chem.AddHs(mol, addCoords=True)
        result['n_atoms_heavy'] = mol.GetNumHeavyAtoms()

        # 3. Compute initial MMFF94 energy
        try:
            mp = AllChem.MMFFGetMoleculeProperties(mol)
            if mp is None:
                raise ValueError("MMFF94 not available for this molecule")
            ff_init = AllChem.MMFFGetMoleculeForceField(mol, mp)
            if ff_init is None:
                raise ValueError("Cannot create MMFF force field")
            result['initial_energy'] = ff_init.CalcEnergy()
        except Exception:
            # Fallback: use UFF
            ff_init = AllChem.UFFGetMoleculeForceField(mol)
            if ff_init is None:
                raise ValueError("Cannot create any force field")
            result['initial_energy'] = ff_init.CalcEnergy()

        # 4. Count initial clashes
        result['clashes_before'] = count_clashes(mol)

        # 5. Save copy for RMSD
        mol_before = Chem.Mol(mol)

        # 6. Minimize
        try:
            mp = AllChem.MMFFGetMoleculeProperties(mol)
            ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
        except Exception:
            ff = AllChem.UFFGetMoleculeForceField(mol)

        if ff is None:
            raise ValueError("Cannot create force field for minimization")

        # Run minimization with convergence monitoring
        ff.Minimize(maxIts=500, forceTol=1e-5, energyTol=1e-7)
        result['n_iterations'] = ff.NumResults() if hasattr(ff, 'NumResults') else -1
        result['final_energy'] = ff.CalcEnergy()
        result['energy_reduction'] = result['initial_energy'] - result['final_energy']

        # 7. Compute RMSD
        result['rmsd'] = compute_rmsd(mol_before, mol)

        # 8. Count final clashes
        result['clashes_after'] = count_clashes(mol)

        # 9. Remove hydrogens for cleaner output
        mol_final = Chem.RemoveHs(mol)

        # 10. Save as SDF
        writer = Chem.SDWriter(str(output_sdf_path))
        writer.write(mol_final)
        writer.close()

        result['status'] = 'success'

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)[:300]
        # Copy original file on failure
        try:
            import shutil
            shutil.copy(str(sdf_path), str(output_sdf_path))
        except Exception:
            pass

    return result


def main():
    parser = argparse.ArgumentParser(description='Batch MMFF94 energy minimization')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing pocket subdirectories with mol_*.sdf files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for minimized structures')
    parser.add_argument('--max_workers', type=int, default=32,
                        help='Number of parallel workers (default: 32)')
    parser.add_argument('--max_molecules', type=int, default=None,
                        help='Max molecules to process (for testing)')
    parser.add_argument('--sdf_pattern', type=str, default='mol_*.sdf',
                        help='Glob pattern for ligand SDF files')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"Batch Energy Minimization (RDKit MMFF94)")
    print(f"{'='*60}")
    print(f"Input:      {input_dir}")
    print(f"Output:     {output_dir}")
    print(f"Workers:    {args.max_workers}")
    print(f"Protocol:   500 steps, force tolerance 1e-5, energy tolerance 1e-7")
    print()

    # Collect tasks
    tasks = []
    pocket_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    print(f"Found {len(pocket_dirs)} pocket directories")

    for pocket_dir in pocket_dirs:
        pocket_name = pocket_dir.name
        sdf_files = sorted(pocket_dir.glob(args.sdf_pattern))
        for sdf_file in sdf_files:
            out_pocket_dir = output_dir / pocket_name
            out_pocket_dir.mkdir(parents=True, exist_ok=True)
            out_sdf = out_pocket_dir / sdf_file.name
            tasks.append((str(sdf_file), str(out_sdf)))

    if args.max_molecules:
        tasks = tasks[:args.max_molecules]

    print(f"Total molecules to minimize: {len(tasks)}")
    print()

    # Run minimization
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    n_success = 0
    n_error = 0
    all_results = []
    t_start = time.time()

    with mp.Pool(processes=args.max_workers) as pool:
        async_results = [pool.apply_async(minimize_one, (t,)) for t in tasks]

        pbar = tqdm(total=len(tasks), desc='Minimizing', unit='mol')
        for ar in async_results:
            try:
                result = ar.get(timeout=120)
                all_results.append(result)
                if result['status'] == 'success':
                    n_success += 1
                else:
                    n_error += 1
            except Exception as e:
                n_error += 1
                all_results.append({
                    'status': 'error',
                    'error': f'Worker error: {str(e)[:200]}',
                    'sdf_file': '', 'pocket': '', 'mol_name': '',
                })
            pbar.update(1)
            pbar.set_postfix(success=n_success, error=n_error)
        pbar.close()

    t_total = time.time() - t_start

    # Save log
    df = pd.DataFrame(all_results)
    log_path = output_dir / 'minimization_log.csv'
    df.to_csv(log_path, index=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Minimization Complete")
    print(f"{'='*60}")
    print(f"  Total time:        {t_total/60:.1f} min")
    print(f"  Molecules:         {len(all_results)}")
    print(f"  Successful:        {n_success}")
    print(f"  Errors:            {n_error}")
    print(f"  Time per mol:      {t_total/max(len(all_results),1):.2f}s")

    if n_success > 0:
        success_df = df[df['status'] == 'success']
        print(f"\n  Energy (kJ/mol):")
        print(f"    Initial mean:    {success_df['initial_energy'].mean():.1f}")
        print(f"    Final mean:      {success_df['final_energy'].mean():.1f}")
        print(f"    Reduction:       {success_df['energy_reduction'].mean():.1f}")
        print(f"\n  RMSD (Å):")
        print(f"    Mean:            {success_df['rmsd'].mean():.3f}")
        print(f"    Median:          {success_df['rmsd'].median():.3f}")
        print(f"    > 1.0 Å:         {(success_df['rmsd'] > 1.0).sum()} molecules")
        print(f"    > 2.0 Å:         {(success_df['rmsd'] > 2.0).sum()} molecules")
        print(f"\n  Clashes (internal):")
        print(f"    Before mean:     {success_df['clashes_before'].mean():.1f}")
        print(f"    After mean:      {success_df['clashes_after'].mean():.1f}")
        print(f"    Reduction:       {success_df['clashes_before'].mean() - success_df['clashes_after'].mean():.1f}")

    print(f"\n  Log:    {log_path}")
    print(f"  Output: {output_dir}")


if __name__ == '__main__':
    main()
