#!/usr/bin/env python3
"""
Physical Validity Validation Script

Computes key physical integrity metrics for generated molecules:
  1. Protein-ligand steric clashes (Clashes/mol, Clash rate)
  2. Bond length anomalies (> 1.7 Å or < 1.0 Å)
  3. Broken aromatic rings

These metrics directly support Main Experiment 2 (§5.3) and the ablation study
(Appendix A), demonstrating that SV-Flow Core maintains physical legality
comparable to DrugFlow baseline while significantly improving diversity.

Usage:
    python scripts/validate_physical.py \
        --samples_dir ./svflow_core_samples \
        --output ./core_physical.csv
"""

import argparse
import sys
import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from tqdm import tqdm

# Add DrugFlow to path for pocket parsing
basedir = Path(__file__).resolve().parent.parent
drugflow_path = Path('/root/baselines/DrugFlow/code/DrugFlow-main')
sys.path.insert(0, str(drugflow_path))
sys.path.insert(0, str(basedir))

from src.constants import atom_decoder, vdw_radii

# Build vdW radii array (matching tangent_projection.py)
_vdw_radii = {**vdw_radii}
_vdw_radii['NH'] = vdw_radii['N']
_vdw_radii['N+'] = vdw_radii['N']
_vdw_radii['O-'] = vdw_radii['O']
_vdw_radii['NOATOM'] = 0.0
VDW_RADII = {a: _vdw_radii.get(a, 1.70) for a in atom_decoder}

# Protein atom name → element mapping (from constants.py)
_PROTEIN_ATOM_TO_ELEMENT = {
    'N': 'N', 'CA': 'C', 'C': 'C', 'O': 'O',
    'CB': 'C', 'CG': 'C', 'CG1': 'C', 'CG2': 'C',
    'CD': 'C', 'CD1': 'C', 'CD2': 'C',
    'CE': 'C', 'CE1': 'C', 'CE2': 'C', 'CE3': 'C',
    'CZ': 'C', 'CZ2': 'C', 'CZ3': 'C', 'CH2': 'C',
    'SG': 'S', 'OG': 'O', 'OG1': 'O',
    'OD1': 'O', 'OD2': 'O', 'OE1': 'O', 'OE2': 'O', 'OH': 'O',
    'ND1': 'N', 'ND2': 'N', 'NE': 'N', 'NE1': 'N', 'NE2': 'N',
    'NH1': 'N', 'NH2': 'N', 'NZ': 'N', 'SD': 'S',
}


# --- Clash detection ---

def parse_pocket_pdb(pdb_path: str) -> dict:
    """Parse a protein PDB file and extract atom coordinates and elements."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pocket', pdb_path)

    coords = []
    elements = []
    for atom in structure.get_atoms():
        coords.append(atom.get_coord())
        atom_name = atom.get_name().strip()
        element = _PROTEIN_ATOM_TO_ELEMENT.get(atom_name, 'C')
        elements.append(element)

    if not coords:
        return {'coords': np.zeros((0, 3)), 'elements': []}

    return {
        'coords': np.array(coords),
        'elements': elements,
    }


def count_clashes(mol, pocket_coords, pocket_elements,
                  tolerance: float = 0.4) -> tuple:
    """
    Count steric clashes between ligand and protein atoms.

    A clash is defined as: distance < (vdW_lig + vdW_pkt - tolerance)

    Returns:
        n_clashes: number of clashing atom pairs
        clash_rate: fraction of ligand atoms involved in clashes
    """
    if mol is None or pocket_coords.shape[0] == 0:
        return 0, 0.0

    try:
        conf = mol.GetConformer()
        lig_coords = conf.GetPositions()  # (N_lig, 3)
    except Exception:
        return 0, 0.0

    n_lig = lig_coords.shape[0]
    if n_lig == 0:
        return 0, 0.0

    # Get ligand vdW radii
    lig_elements = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        lig_elements.append(sym if sym in VDW_RADII else 'C')
    lig_radii = np.array([VDW_RADII[e] for e in lig_elements])
    pkt_radii = np.array([VDW_RADII.get(e, 1.70) for e in pocket_elements])

    n_clashes = 0
    n_lig_clash = set()

    # Compute pairwise distances and check clashes
    for i in range(n_lig):
        diff = lig_coords[i] - pocket_coords  # (N_pkt, 3)
        dist = np.sqrt((diff ** 2).sum(axis=1))
        vdw_sum = lig_radii[i] + pkt_radii
        clash_idx = np.where(dist < (vdw_sum - tolerance))[0]
        n_clashes += len(clash_idx)
        if len(clash_idx) > 0:
            n_lig_clash.add(i)

    clash_rate = len(n_lig_clash) / n_lig if n_lig > 0 else 0.0
    return n_clashes, clash_rate


# --- Bond anomaly detection ---

def count_bond_anomalies(mol, max_bond: float = 1.7, min_bond: float = 1.0) -> tuple:
    """
    Count bond length anomalies in a molecule.

    Returns:
        n_anomalies: number of bonds outside [min_bond, max_bond]
        anomaly_rate: fraction of anomalous bonds
    """
    if mol is None:
        return 0, 0.0

    try:
        conf = mol.GetConformer()
        coords = conf.GetPositions()
    except Exception:
        return 0, 0.0

    n_bonds = mol.GetNumBonds()
    if n_bonds == 0:
        return 0, 0.0

    n_anomalies = 0
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        length = np.linalg.norm(coords[i] - coords[j])
        if length > max_bond or length < min_bond:
            n_anomalies += 1

    anomaly_rate = n_anomalies / n_bonds
    return n_anomalies, anomaly_rate


# --- Broken ring detection ---

def count_broken_rings(mol, max_ring_bond: float = 2.0) -> int:
    """
    Count broken aromatic/small rings in a molecule.

    A ring is considered broken if any bond in the ring exceeds max_ring_bond Å.
    We check the SSSR (Smallest Set of Smallest Rings).
    """
    if mol is None:
        return 0

    try:
        conf = mol.GetConformer()
        coords = conf.GetPositions()
    except Exception:
        return 0

    ssr = Chem.GetSymmSSSR(mol)
    n_broken = 0

    for ring in ssr:
        ring_atoms = list(ring)
        if len(ring_atoms) < 3:
            continue
        # Check each consecutive pair in the ring
        for k in range(len(ring_atoms)):
            i = ring_atoms[k]
            j = ring_atoms[(k + 1) % len(ring_atoms)]
            length = np.linalg.norm(coords[i] - coords[j])
            if length > max_ring_bond:
                n_broken += 1
                break  # count each ring only once

    return n_broken


# --- Main validation ---

def validate_directory(samples_dir: Path) -> list:
    """Validate all molecules in a directory of pocket subdirectories."""
    pocket_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir()])
    print(f'Validating {len(pocket_dirs)} pocket directories...')

    results = []
    for pocket_dir in tqdm(pocket_dirs, desc='Validating'):
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
            continue

        # Load pocket PDB for clash detection
        pocket_files = list(pocket_dir.glob('*.pdb'))
        pocket_data = {'coords': np.zeros((0, 3)), 'elements': []}
        if pocket_files:
            try:
                pocket_data = parse_pocket_pdb(str(pocket_files[0]))
            except Exception:
                pass

        # Compute metrics for each molecule
        total_clashes = 0
        total_anomalies = 0
        total_bonds = 0
        total_broken_rings = 0
        total_lig_atoms = 0

        for mol in mols:
            n_clashes, clash_rate = count_clashes(
                mol, pocket_data['coords'], pocket_data['elements']
            )
            total_clashes += n_clashes

            n_anomalies, anomaly_rate = count_bond_anomalies(mol)
            total_anomalies += n_anomalies

            n_bonds = mol.GetNumBonds()
            total_bonds += n_bonds

            n_broken = count_broken_rings(mol)
            total_broken_rings += n_broken

            total_lig_atoms += mol.GetNumAtoms()

        n_mols = len(mols)
        results.append({
            'pocket': pocket_name,
            'n_molecules': n_mols,
            'clashes_per_mol': total_clashes / n_mols if n_mols > 0 else 0.0,
            'clash_rate': total_clashes / total_lig_atoms if total_lig_atoms > 0 else 0.0,
            'bond_anomalies_per_mol': total_anomalies / n_mols if n_mols > 0 else 0.0,
            'bond_anomaly_rate': total_anomalies / total_bonds if total_bonds > 0 else 0.0,
            'broken_rings_per_mol': total_broken_rings / n_mols if n_mols > 0 else 0.0,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description='Physical Validity Validation')
    parser.add_argument('--samples_dir', type=str, required=True,
                        help='Directory containing generated molecule subdirectories')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file')
    args = parser.parse_args()

    warnings.filterwarnings('ignore')

    results = validate_directory(Path(args.samples_dir))

    if not results:
        print('No valid molecules found!')
        return

    # Summary statistics
    df = pd.DataFrame(results)
    print(f'\n{"="*60}')
    print(f'Physical Validity Summary ({len(results)} pockets)')
    print(f'{"="*60}')

    for col in ['clashes_per_mol', 'clash_rate', 'bond_anomalies_per_mol',
                'bond_anomaly_rate', 'broken_rings_per_mol']:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            print(f'  {col:30s}: {mean_val:.3f} ± {std_val:.3f}')

    if args.output:
        df.to_csv(args.output, index=False)
        print(f'\nResults saved to {args.output}')


if __name__ == '__main__':
    main()
