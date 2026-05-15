"""Non-neural baselines for thesis comparison.

The thesis compares the learned model against:

1. ``always-<k>``         constant policy that always picks solver k.
2. ``oracle``             the unrealistic optimum (lower bound on regret).
3. ``linear``             ridge regression in log-time space, one model
                          per solver. Closed-form solution; no NN needed.
4. ``nearest_neighbor``   1-NN in standardised feature space; predicts
                          the solver that won on the closest training row.

Each returns a callable ``predict_log_t(features) -> Tensor (B, S)``.
At evaluation time the same ``evaluate()`` framework consumes these,
so every baseline appears on the same table as the neural models.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BaselineResult:
    name: str
    pred_log_t: torch.Tensor


def always_solver(num_samples: int, num_solvers: int, k: int) -> torch.Tensor:
    log_t = torch.full((num_samples, num_solvers), 1.0)
    log_t[:, k] = -1.0  # encourage argmin to pick k
    return log_t


def oracle(true_t: torch.Tensor) -> torch.Tensor:
    """Returns the ground-truth log-time so ``argmin`` is the optimum."""
    return true_t.clamp_min(1e-12).log()


def ridge_per_solver(
    x_train: torch.Tensor,
    log_t_train: torch.Tensor,
    x_val: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Fit one ridge regression per solver in log-time space.

    Standard analytic solution; this is the same model used as a tabular
    baseline in autotuning papers.
    """
    # Standardise inputs.
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-8)
    Xtr = (x_train - mean) / std
    Xtr = torch.cat([Xtr, torch.ones(Xtr.size(0), 1)], dim=1)  # bias
    Xval = (x_val - mean) / std
    Xval = torch.cat([Xval, torch.ones(Xval.size(0), 1)], dim=1)

    d = Xtr.size(1)
    A = Xtr.T @ Xtr + alpha * torch.eye(d)
    B = Xtr.T @ log_t_train
    W = torch.linalg.solve(A, B)  # (d, S)
    return Xval @ W


def nearest_neighbor(
    x_train: torch.Tensor,
    log_t_train: torch.Tensor,
    x_val: torch.Tensor,
) -> torch.Tensor:
    """1-NN: return the training neighbor's solver times.

    Distance in standardised feature space.
    """
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-8)
    Xtr = (x_train - mean) / std
    Xval = (x_val - mean) / std

    # Pairwise distances; for the dataset sizes we expect (~10k) this
    # fits in memory comfortably. Switch to chunked if you scale up.
    d2 = torch.cdist(Xval, Xtr, p=2)
    idx = d2.argmin(dim=1)
    return log_t_train[idx]
