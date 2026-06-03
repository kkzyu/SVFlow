"""
Time-Annealed Scheduling for SV-Flow Guidance (§4.5)

Late-onset strategy: guidance is disabled during early denoising (t > t_on)
and smoothly ramped up in the later stages using a parabolic profile.

λ(t) = 0,                  for t > t_on
λ(t) = λ_max * (1 - t/t_on)^2,  for t <= t_on
"""

import torch


class TimeAnnealedScheduler:
    """
    Late-onset time-annealing scheduler for SV-Flow guidance strength.

    In early denoising (t near 1), the predicted clean molecule x_hat_0
    has high uncertainty; applying SVGD gradients would introduce artifacts.
    The scheduler disables guidance in the first half of sampling and smoothly
    ramps it up in the second half.

    Default: t_on = 0.5, lambda_max = 1.0
    """

    def __init__(self, t_on: float = 0.5, lambda_max: float = 1.0):
        if not 0.0 < t_on <= 1.0:
            raise ValueError(f"t_on must be in (0, 1], got {t_on}")
        self.t_on = t_on
        self.lambda_max = lambda_max

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute guidance strength λ at time t.

        Args:
            t: scalar or tensor of current ODE time in [0, 1]
               where t=1 is pure noise, t=0 is data

        Returns:
            lambda: guidance strength, same shape as t
        """
        # t=1 is noise, t=0 is data
        # Guidance enabled when t <= t_on
        lam = torch.where(
            t <= self.t_on,
            self.lambda_max * (1.0 - t / self.t_on) ** 2,
            torch.zeros_like(t)
        )
        return lam
