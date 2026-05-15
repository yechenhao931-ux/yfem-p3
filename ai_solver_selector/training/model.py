"""Solver-time regression network shared by training and TorchScript export.

The network takes a feature vector (see matrix_features.hpp) of length
``num_features`` and outputs one predicted runtime per solver
(see solver_registry.hpp), length ``num_solvers``. At inference time the C++
side picks ``argmin`` over the outputs to choose the best solver.

Two architectures are provided:

* ``SolverSelector`` -  a straightforward MLP (the thesis baseline).
* ``MoESolverSelector`` -  a shared trunk plus one head per problem type.
  This is a mixture-of-experts variant; the appropriate head is selected
  by the first feature (``problem_type``). It captures type-specific
  spectral patterns (e.g. convection-dominated systems behave very
  differently from pure mass solves) while still sharing low-level
  feature representations across all problems.

Both models predict in *log-time* space because solver runtimes span
several orders of magnitude; with raw seconds the regression collapses
onto the largest examples. The C++ side exponentiates and picks
``argmin``.

Dropout is built into both models so the trained network can be queried
with Monte Carlo dropout at inference time for uncertainty estimation.
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn


def _mlp_block(in_dim: int, hidden: int, depth: int, dropout: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    d = in_dim
    for _ in range(depth):
        layers.append(nn.Linear(d, hidden))
        layers.append(nn.GELU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        d = hidden
    return nn.Sequential(*layers)


class SolverSelector(nn.Module):
    """Baseline MLP regressor: features -> log-time per solver."""

    def __init__(
        self,
        num_features: int,
        num_solvers: int,
        hidden: int = 64,
        depth: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.trunk = _mlp_block(num_features, hidden, depth, dropout)
        self.head = nn.Linear(hidden, num_solvers)

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
        z = (x - self.feat_mean) / self.feat_std
        y = self.head(self.trunk(z))
        return y * self.time_log_std + self.time_log_mean


class MoESolverSelector(nn.Module):
    """Mixture-of-experts: shared trunk, per-problem-type head.

    ``problem_type`` is expected as the *first* feature; this matches the
    layout in ``matrix_features.hpp``. The head index is clamped to the
    range of known types so out-of-distribution problem ids still get a
    valid prediction (from the closest known head).
    """

    def __init__(
        self,
        num_features: int,
        num_solvers: int,
        num_problem_types: int,
        hidden: int = 64,
        depth: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_problem_types = num_problem_types
        self.trunk = _mlp_block(num_features, hidden, depth, dropout)
        self.heads = nn.ModuleList(
            [nn.Linear(hidden, num_solvers) for _ in range(num_problem_types)]
        )

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
        z = (x - self.feat_mean) / self.feat_std
        h = self.trunk(z)
        # Per-row head selection: stack all heads, gather the row's head.
        # This keeps the path differentiable and TorchScript-friendly.
        head_outs = torch.stack([head(h) for head in self.heads], dim=1)
        # problem_type is feature 0 in raw space; round to nearest int.
        pt = x[:, 0].round().long().clamp(0, self.num_problem_types - 1)
        gather_idx = pt.view(-1, 1, 1).expand(-1, 1, head_outs.size(-1))
        y = head_outs.gather(1, gather_idx).squeeze(1)
        return y * self.time_log_std + self.time_log_mean


@torch.no_grad()
def predict_with_uncertainty(
    model: nn.Module, x: torch.Tensor, n_samples: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Monte Carlo dropout: keep dropout layers in train mode at inference.

    Returns ``(mean_log_t, std_log_t)`` where ``std_log_t`` measures the
    model's epistemic uncertainty per solver, per sample. Useful for
    falling back to a safe default when the network is unsure.
    """
    was_training = model.training
    model.train()
    # Freeze everything except dropout: walk modules, switch BN/LN to eval.
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
            m.eval()

    preds = torch.stack([model(x) for _ in range(n_samples)], dim=0)
    mean = preds.mean(dim=0)
    std = preds.std(dim=0)

    if not was_training:
        model.eval()
    return mean, std
