"""
Kinematic Decoupling of Molecular Velocity Fields (§4.1)

Decomposes the full atomistic velocity field from the base flow matching model
into internal conformational velocity (v_int) and center-of-mass translational
velocity (v_CoM). All SV-Flow guidance operates only on v_CoM.
"""

import torch


def compute_center_of_mass(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the center of mass for each molecule in a batch.

    Args:
        x: atom coordinates, shape (N_atoms_total, 3)
        mask: batch assignment mask, shape (N_atoms_total,)
              mask[i] = molecule_id for atom i

    Returns:
        com: center of mass per molecule, shape (N_molecules, 3)
    """
    from torch_scatter import scatter_mean
    return scatter_mean(x, mask, dim=0)


def decompose_velocity(vel: torch.Tensor, mask: torch.Tensor) -> tuple:
    """
    Decompose the full atomistic velocity field v_θ into:
      v_int: internal conformational velocity (zero CoM motion)
      v_CoM: translational velocity (same for all atoms of a molecule)

    Args:
        vel: full velocity field, shape (N_atoms_total, 3)
        mask: batch assignment mask, shape (N_atoms_total,)

    Returns:
        v_int: internal velocity, shape (N_atoms_total, 3)
        v_CoM: translational velocity per atom, shape (N_atoms_total, 3)
    """
    from torch_scatter import scatter_mean

    v_CoM_per_mol = scatter_mean(vel, mask, dim=0)          # (N_mols, 3)
    v_CoM = v_CoM_per_mol[mask]                              # broadcast to atoms
    v_int = vel - v_CoM
    return v_int, v_CoM


def decompose_velocity_by_molecule(
    vel: torch.Tensor, sizes: list
) -> tuple:
    """
    Decompose velocity when molecules are concatenated without a mask.
    Each molecule's atoms are contiguous.

    Args:
        vel: full velocity field, shape (sum(sizes), 3)
        sizes: list of atom counts per molecule, e.g. [15, 12, 18]

    Returns:
        v_int: internal velocity, shape (sum(sizes), 3)
        v_CoM: translational velocity per atom, shape (sum(sizes), 3)
        v_CoM_per_mol: translational velocity per molecule, shape (N_mols, 3)
    """
    v_CoM_per_mol = torch.stack([v.mean(dim=0) for v in torch.split(vel, sizes)])
    v_CoM = torch.cat([vcm.expand(s, -1) for vcm, s in zip(v_CoM_per_mol, sizes)])
    v_int = vel - v_CoM
    return v_int, v_CoM, v_CoM_per_mol


def broadcast_com_to_atoms(v_com: torch.Tensor, sizes: list) -> torch.Tensor:
    """
    Broadcast per-molecule CoM-space vectors to all atoms.

    Args:
        v_com: per-molecule vectors in CoM space, shape (N_mols, 3)
        sizes: list of atom counts per molecule

    Returns:
        Broadcasted vectors, shape (sum(sizes), 3)
    """
    return torch.cat([v.expand(s, -1) for v, s in zip(v_com, sizes)])


def compute_ligand_com_per_molecule(
    x: torch.Tensor, sizes: list
) -> torch.Tensor:
    """
    Compute center of mass for each molecule.

    Args:
        x: all atom coordinates concatenated, shape (sum(sizes), 3)
        sizes: list of atom counts per molecule

    Returns:
        com: center of mass per molecule, shape (N_mols, 3)
    """
    return torch.stack([xi.mean(dim=0) for xi in torch.split(x, sizes)])
