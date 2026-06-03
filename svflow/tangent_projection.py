"""
Tangent Plane Projection in CoM Space (§4.3)

Projects SVGD repulsive velocities onto the protein surface tangent plane
to prevent molecules from "crashing through" the protein backbone.

The protein surface normal is estimated from the gradient of a soft vdW
repulsive energy (vdW clash penalty) w.r.t. the ligand center of mass.

Key insight: the gradient ∇_c E_protein points from the protein interior
to the exterior (the normal direction). By removing the component of the
SVGD velocity along this normal, we constrain molecular motion to slide
along the protein surface.
"""

import torch
from torch_scatter import scatter_mean

from src.constants import atom_decoder, vdw_radii


# Build vdW radii array matching the DrugFlow atom type ordering
_vdw_radii = {**vdw_radii}
_vdw_radii['NH'] = vdw_radii['N']
_vdw_radii['N+'] = vdw_radii['N']
_vdw_radii['O-'] = vdw_radii['O']
_vdw_radii['NOATOM'] = 0
VDW_RADII_ARRAY = torch.tensor([_vdw_radii[a] for a in atom_decoder])


def compute_protein_vdw_gradient(
    ligand_coords: torch.Tensor,
    ligand_types: torch.Tensor,
    ligand_mask: torch.Tensor,
    pocket_coords: torch.Tensor,
    pocket_types: torch.Tensor,
    pocket_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the gradient of protein-ligand vdW clash energy w.r.t. ligand atoms.

    Uses a soft vdW penalty: max(0, 1 - dist / (r_i + r_j)).
    This is a simplified, differentiable approximation to the true vdW repulsion.

    Args:
        ligand_coords: ligand atom coordinates, shape (N_lig, 3)
        ligand_types: integer atom types, shape (N_lig,)
        ligand_mask: batch mask for ligands, shape (N_lig,)
        pocket_coords: protein atom coordinates, shape (N_pkt, 3)
        pocket_types: protein atom types (integers), shape (N_pkt,)
        pocket_mask: batch mask for pocket, shape (N_pkt,)

    Returns:
        grad_per_atom: gradient w.r.t. each ligand atom, shape (N_lig, 3)
    """
    device = ligand_coords.device

    lig_radii = VDW_RADII_ARRAY.to(device)[ligand_types]
    pkt_radii = VDW_RADII_ARRAY.to(device)[pocket_types]

    diff = ligand_coords[:, None, :] - pocket_coords[None, :, :]  # (N_lig, N_pkt, 3)
    dist = torch.norm(diff, dim=-1)                                  # (N_lig, N_pkt)
    sum_vdw = lig_radii[:, None] + pkt_radii[None, :]                # (N_lig, N_pkt)

    # Clash condition: dist < sum_vdw
    # For each clashing pair, gradient contribution: -(1/sum_vdw) * diff_unit
    # This comes from: d/dx max(0, 1 - dist/sum_vdw) = -(1/sum_vdw) * (x_i - x_j)/dist
    clash_mask = (dist < sum_vdw) & (dist > 1e-6)

    diff_unit = diff / (dist.unsqueeze(-1) + 1e-8)
    grad_per_pair = -diff_unit / (sum_vdw.unsqueeze(-1) + 1e-8)
    grad_per_pair = grad_per_pair * clash_mask.unsqueeze(-1).float()

    # Sum over all pocket atoms
    grad_per_atom = grad_per_pair.sum(dim=1)  # (N_lig, 3)

    return grad_per_atom


def compute_com_vdw_gradient(
    ligand_coords: torch.Tensor,
    ligand_types: torch.Tensor,
    ligand_mask: torch.Tensor,
    pocket_coords: torch.Tensor,
    pocket_types: torch.Tensor,
    pocket_mask: torch.Tensor,
    ligand_sizes: list,
) -> torch.Tensor:
    """
    Compute protein vdW gradient aggregated to CoM level.

    Each molecule's per-atom vdW gradients are averaged to get the
    force on the center of mass: ∇_c E_protein = mean(per-atom gradient).

    Args:
        ligand_coords: all ligand atoms, shape (sum(sizes), 3)
        ligand_types: atom types, shape (sum(sizes),)
        ligand_mask: batch mask, shape (sum(sizes),)
        pocket_coords: protein atoms, shape (N_pkt, 3)
        pocket_types: protein atom types, shape (N_pkt,)
        pocket_mask: pocket batch mask, shape (N_pkt,)
        ligand_sizes: list of atom counts per molecule

    Returns:
        grad_com: gradient w.r.t. CoM per molecule, shape (N_mols, 3)
    """
    grad_per_atom = compute_protein_vdw_gradient(
        ligand_coords, ligand_types, ligand_mask,
        pocket_coords, pocket_types, pocket_mask
    )

    # Aggregate to CoM: the force on CoM = mean of per-atom forces
    grad_com = torch.stack([
        g.mean(dim=0) for g in torch.split(grad_per_atom, ligand_sizes)
    ])

    return grad_com


def tangent_plane_projection(
    delta_v_svgd: torch.Tensor,
    com_gradient: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Project SVGD velocity onto the tangent plane of the protein surface.

    ΔV_proj = ΔV_svgd - (ΔV_svgd · n̂) n̂

    where n̂ = ∇_c E_protein / ||∇_c E_protein|| is the unit surface normal,
    pointing from the protein interior to exterior.

    This removes the component that would push molecules through the protein,
    allowing only surface-tangent sliding motion.

    Args:
        delta_v_svgd: SVGD velocity vectors, shape (N, 3)
        com_gradient: protein vdW gradient at CoM, shape (N, 3)
        eps: numerical stability

    Returns:
        delta_v_proj: projected velocity, shape (N, 3)
    """
    grad_norm = torch.norm(com_gradient, dim=-1, keepdim=True)
    # If the gradient is zero (no clashes), use zero normal -> no projection needed
    normal = torch.where(
        grad_norm > eps,
        com_gradient / grad_norm,
        torch.zeros_like(com_gradient)
    )

    # Remove the normal component
    normal_component = (delta_v_svgd * normal).sum(dim=-1, keepdim=True) * normal
    delta_v_proj = delta_v_svgd - normal_component

    return delta_v_proj
