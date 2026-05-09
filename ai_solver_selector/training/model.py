"""Solver-time regression network shared by training and TorchScript export.

The network takes a feature vector (see matrix_features.hpp) of length
``num_features`` and outputs one predicted runtime per solver
(see solver_registry.hpp), length ``num_solvers``. At inference time the C++
side picks ``argmin`` over the outputs to choose the best solver.

We deliberately keep the network small so it loads instantly inside an MFEM
solve and so TorchScript export stays trivial. Predictions are made in
log-time space because solver runtimes span several orders of magnitude.
"""

from __future__ import annotations

import torch
from torch import nn


class SolverSelector(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_solvers: int,
        hidden: int = 64,
        depth: int = 3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.GELU())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, num_solvers))
        self.net = nn.Sequential(*layers)

        self.register_buffer("feat_mean", torch.zeros(num_features))
        self.register_buffer("feat_std", torch.ones(num_features))
        self.register_buffer("time_log_mean", torch.zeros(num_solvers))
        self.register_buffer("time_log_std", torch.ones(num_solvers))

    def set_normalization(
        self,
        feat_mean: torch.Tensor,
        feat_std: torch.Tensor,
        time_log_mean: torch.Tensor,
        time_log_std: torch.Tensor,
    ) -> None:
        self.feat_mean.copy_(feat_mean)
        self.feat_std.copy_(feat_std.clamp_min(1e-8))
        self.time_log_mean.copy_(time_log_mean)
        self.time_log_std.copy_(time_log_std.clamp_min(1e-8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize features, predict normalized log-times, then de-normalize
        # back into log-time space. The C++ side does ``argmin`` on this.
        z = (x - self.feat_mean) / self.feat_std
        y = self.net(z)
        return y * self.time_log_std + self.time_log_mean
