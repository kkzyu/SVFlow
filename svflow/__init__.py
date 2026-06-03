"""
SV-Flow: Stein Variational Flow Matching for Diverse Structure-Based Drug Design.

Simplified architecture (post-ablation):
  - kinematics: kinematic decoupling of velocity fields (§4.1)
  - svgd: SVGD kernel and repulsive interaction field in CoM space (§4.2)
  - time_scheduler: time-annealed scheduling λ(t) (§4.3)
  - sampler: multi-trajectory SV-Flow sampling orchestrator (§4.4)

Note: tangent_projection (§4.3-old) and orthogonal_preservation (§4.4-old)
are retained for ablation studies only. They are NOT part of the recommended
SV-Flow Core configuration, as systematic ablation demonstrated they severely
suppress spatial diversity.
"""

from svflow.sampler import SVFlowSampler
from svflow.svgd import (
    compute_svgd_velocity,
    compute_isotropic_repulsion,
    rbf_kernel,
    median_heuristic,
)
from svflow.kinematics import decompose_velocity, compute_center_of_mass
from svflow.time_scheduler import TimeAnnealedScheduler
