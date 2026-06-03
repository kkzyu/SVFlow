"""
Stein Variational Gradient Descent (SVGD) Kernel and Repulsive Field (§4.2)

Implements the RBF kernel with median heuristic bandwidth and the SVGD
repulsive velocity computation in CoM space (R^3) for multi-trajectory
diversity-driven sampling.
"""

import torch


def median_heuristic(distances: torch.Tensor) -> torch.Tensor:
    """
    Compute the median heuristic bandwidth for the RBF kernel.

    h = median(||c_i - c_j||^2) / log(N)
    where N is the number of particles.

    Args:
        distances: squared pairwise distances, shape (N, N)

    Returns:
        h: kernel bandwidth (scalar)
    """
    N = distances.shape[0]
    if N <= 1:
        return torch.tensor(1.0, device=distances.device)
    med_sq = torch.median(distances[distances > 0]) if (distances > 0).any() else torch.tensor(1.0, device=distances.device)
    return med_sq / torch.log(torch.tensor(N, dtype=torch.float32, device=distances.device))


def rbf_kernel(x: torch.Tensor, h: float = None) -> torch.Tensor:
    """
    Compute the RBF (Gaussian) kernel matrix.

    K_ij = exp(-||x_i - x_j||^2 / (2 * h))

    Args:
        x: particle positions, shape (N, D)
        h: bandwidth (scalar). If None, uses median heuristic.

    Returns:
        K: kernel matrix, shape (N, N)
        sq_dists: squared pairwise distances, shape (N, N)
    """
    sq_dists = torch.cdist(x, x, p=2).pow(2)
    if h is None:
        h = median_heuristic(sq_dists)
    K = torch.exp(-sq_dists / (2.0 * h + 1e-8))
    return K, sq_dists


def rbf_kernel_and_grad(x: torch.Tensor, h: float = None):
    """
    Compute RBF kernel matrix and its gradient w.r.t. first argument.

    K(x_i, x_j) = exp(-||x_i - x_j||^2 / (2h))
    ∇_{x_j} K(x_i, x_j) = K(x_i, x_j) * (x_i - x_j) / h

    Args:
        x: particle positions, shape (N, D)
        h: bandwidth. If None, uses median heuristic.

    Returns:
        K: kernel matrix, shape (N, N)
        grad_K: per-pair kernel gradients, shape (N, N, D)
            grad_K[i, j] = ∇_{x_j} k(x_i, x_j)
    """
    N, D = x.shape
    sq_dists = torch.cdist(x, x, p=2).pow(2)
    if h is None:
        h = median_heuristic(sq_dists)
    K = torch.exp(-sq_dists / (2.0 * h + 1e-8))

    diff = x.unsqueeze(0) - x.unsqueeze(1)     # (N, N, D): x_j - x_i
    grad_K = K.unsqueeze(-1) * (-diff) / (h + 1e-8)   # ∇_{x_j} k(x_i, x_j)

    return K, grad_K, h


def compute_repulsive_energy_gradient(
    com_positions: torch.Tensor,
    d_min: float = 2.0
) -> torch.Tensor:
    """
    Compute the gradient of the repulsive energy E_rep w.r.t. CoM positions.

    E_rep(c) = -ln(d_min) approximately, using a soft repulsive potential.
    We use: E_rep(c_i, c_j) = -ln(||c_i - c_j||) for ||c_i - c_j|| < d_min,
    and 0 beyond that threshold (to prevent long-range spurious forces).

    ∇_{c_j} E_rep(c_j) is computed as the gradient w.r.t. each particle's position
    from the total repulsive energy.

    Args:
        com_positions: CoM positions, shape (N, 3)
        d_min: distance threshold below which repulsion is active

    Returns:
        grad_E: gradient w.r.t. each position, shape (N, 3)
    """
    N = com_positions.shape[0]
    if N <= 1:
        return torch.zeros_like(com_positions)

    diff = com_positions.unsqueeze(0) - com_positions.unsqueeze(1)   # (N, N, 3): c_i - c_j
    dist = torch.norm(diff, dim=-1)                                    # (N, N)

    # Soft repulsive potential: -ln(dist) with cutoff
    # Gradient: dr/dc * dE/dr = (c_i - c_j)/dist^2 * 1 (attractive toward larger dist)
    # For repulsion: we push AWAY from others, so the force on j from i is:
    # ∇_{c_j} E_rep = (c_j - c_i) / (dist^2 + eps) for dist < d_min

    mask = (dist < d_min) & (dist > 1e-6)
    inv_dist_sq = 1.0 / (dist.pow(2) + 1e-8)
    grad_per_pair = diff * inv_dist_sq.unsqueeze(-1)   # (N, N, 3): direction from j to i / dist

    grad_per_pair = grad_per_pair * mask.unsqueeze(-1).float()

    # Sum over all j for each i: gradient = sum_j (c_i - c_j) / dist^2
    grad_E = grad_per_pair.sum(dim=1)  # (N, 3)

    return grad_E


def compute_isotropic_repulsion(
    com_positions: torch.Tensor,
    d_min: float = 2.0,
    power: float = 2.0,
) -> torch.Tensor:
    """
    Compute isotropic 1/r^power distance repulsion in CoM space.

    This is a simplified Metadiffusion-style repulsion: each particle is pushed
    away from every other particle with a force proportional to 1/r^power.

    ΔV_iso^{(i)} = Σ_{j≠i} (c_i - c_j) / ||c_i - c_j||^{power+1}

    Unlike SVGD, this lacks the Stein operator (entropy maximization) term
    and the adaptive RBF kernel weighting, making it less effective at
    adapting to irregularly shaped protein pockets.

    Args:
        com_positions: CoM positions, shape (N, 3)
        d_min: distance threshold below which repulsion is active
        power: exponent for distance decay (default 2.0 → 1/r²)

    Returns:
        delta_V_iso: isotropic repulsion velocity, shape (N, 3)
    """
    N = com_positions.shape[0]
    if N <= 1:
        return torch.zeros_like(com_positions)

    diff = com_positions.unsqueeze(0) - com_positions.unsqueeze(1)   # (N, N, 3)
    dist = torch.norm(diff, dim=-1) + 1e-8                            # (N, N)

    # Only apply repulsion within d_min
    mask = (dist < d_min) & (dist > 1e-6)
    inv_dist_pow = 1.0 / (dist.pow(power + 1) + 1e-8)
    force = diff * inv_dist_pow.unsqueeze(-1)                          # (N, N, 3)
    force = force * mask.unsqueeze(-1).float()

    delta_V = force.sum(dim=1)  # (N, 3)
    return delta_V / N


def compute_svgd_velocity(
    com_positions: torch.Tensor,
    d_min: float = 2.0,
    h: float = None
) -> torch.Tensor:
    """
    Compute the SVGD repulsive translational velocity increment for each particle.

    ΔV_{SVGD}^{(i)} = (1/N) * Σ_{j≠i} [
        k(c_j, c_i) * ∇_{c_j} E_rep(c_j)    (kernel-weighted repulsion)
        + ∇_{c_j} k(c_j, c_i)               (kernel gradient / Stein operator)
    ]

    This is a pure R^3 vector per molecule, operating only on translational CoM
    motion. The kernel gradient term is the Stein operator that drives the system
    toward maximum Shannon entropy.

    Args:
        com_positions: predicted clean CoM positions, shape (N, 3)
        d_min: distance threshold for repulsive energy
        h: RBF kernel bandwidth. If None, uses median heuristic.

    Returns:
        delta_V_svgd: SVGD velocity in CoM space, shape (N, 3)
    """
    N = com_positions.shape[0]
    if N <= 1:
        return torch.zeros_like(com_positions)

    # RBF kernel and its gradient
    K, grad_K, h = rbf_kernel_and_grad(com_positions, h)

    # Repulsive energy gradient
    grad_E = compute_repulsive_energy_gradient(com_positions, d_min)

    # Term 1: kernel-weighted repulsion (weighted by K[j, i])
    # For particle i: Σ_j K(c_j, c_i) * ∇_{c_j} E_rep
    # K_ji = K[j, i]
    weighted_repulsion = K.t() @ grad_E  # (N, D): K_ji * grad_E_j

    # Term 2: kernel gradient (Stein operator)
    # For particle i: Σ_j ∇_{c_j} k(c_j, c_i)
    # grad_K[j, i] = ∇_{c_i} k(c_j, c_i) — gradient of kernel w.r.t. c_i
    # We need grad_K indexed so that grad_K[j, i] gives the gradient w.r.t. c_i
    # In rbf_kernel_and_grad, grad_K[i, j] = ∇_{c_j} k(c_i, c_j)
    # So grad_K[j, i] = ∇_{c_i} k(c_j, c_i) -- transpose first two dims
    kernel_gradient_term = grad_K.sum(dim=0)  # Σ_j ∇_{c_i} k(c_j, c_i)

    delta_V = (weighted_repulsion + kernel_gradient_term) / N

    return delta_V
