"""
Orthogonal Kinetic Preservation (§4.4)

Preserves the base model's optimal transport (OT) low-KPE manifold by
projecting the guidance velocity onto the subspace orthogonal to the
original flow matching velocity direction.

ΔV_⊥ = ΔV_proj - (ΔV_proj · v̂_CoM) v̂_CoM

where v̂_CoM = v_CoM / ||v_CoM|| is the unit direction of the base model's
translational velocity.

This ensures ||v_CoM|| remains unchanged — the guidance only rotates the
direction, not the magnitude, of the translational component.
"""

import torch


def orthogonal_preservation(
    delta_v_proj: torch.Tensor,
    v_com: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Project guidance velocity onto the subspace orthogonal to v_CoM.

    Args:
        delta_v_proj: tangent-projected SVGD velocity, shape (N, 3)
        v_com: base model's CoM velocity per molecule, shape (N, 3)
        eps: numerical stability

    Returns:
        delta_v_orth: orthogonal-preserved velocity, shape (N, 3)
    """
    v_com_norm = torch.norm(v_com, dim=-1, keepdim=True)

    # Unit direction of v_CoM
    v_com_hat = torch.where(
        v_com_norm > eps,
        v_com / v_com_norm,
        torch.zeros_like(v_com)
    )

    # Remove the component parallel to v_CoM
    parallel_component = (delta_v_proj * v_com_hat).sum(dim=-1, keepdim=True) * v_com_hat
    delta_v_orth = delta_v_proj - parallel_component

    return delta_v_orth


def compute_kpe_contribution(
    v_com: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """
    Compute the instantaneous kinetic path energy contribution from CoM motion.

    d(KPE) = ||v_CoM||^2 * dt

    Useful for monitoring KPE during sampling.

    Args:
        v_com: CoM velocity, shape (N, 3)
        dt: time step size (scalar)

    Returns:
        d_kpe: KPE contribution for this step, shape (N,)
    """
    return (v_com ** 2).sum(dim=-1) * dt
