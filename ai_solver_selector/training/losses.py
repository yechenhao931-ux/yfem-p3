"""Loss functions for cost-aware solver selection.

Three losses are provided. They optimize complementary objectives, and
the experimental section of the thesis compares them on the same data.

* ``regression_loss``
    Smooth-L1 in log-time space, one scalar per (sample, solver) pair.
    Predicts the actual runtime. This is the standard regression
    baseline used in the literature on autotuning. It minimizes
    *fitting error* but not *decision error*.

* ``regret_loss``
    Lets the network minimize the *selection regret* directly. Given
    predicted log-times ``y_hat`` and ground-truth times ``t``, we form
    soft-selection weights ``w = softmax(-y_hat / tau)`` and compute the
    expected time under those weights. The loss is the relative regret
    ``(E_w[t] - min(t)) / min(t)``. As ``tau -> 0`` this approaches the
    true regret of the argmin pick; at moderate ``tau`` it provides
    useful gradient signal to all solvers.

* ``listwise_loss``
    ListMLE / Plackett-Luce: the negative log-likelihood of the true
    ranking of solvers (slowest -> fastest) under the predicted scores.
    Optimizes the *ordering* over solvers and is robust to absolute
    miscalibration.
"""

from __future__ import annotations

import torch
from torch import nn


def regression_loss(
    pred_log_t: torch.Tensor, true_log_t: torch.Tensor, **_: object
) -> torch.Tensor:
    return nn.functional.smooth_l1_loss(pred_log_t, true_log_t)


def regret_loss(
    pred_log_t: torch.Tensor,
    true_log_t: torch.Tensor,
    true_t: torch.Tensor | None = None,
    tau: float = 0.1,
    clip_regret: float = 10.0,
    **_: object,
) -> torch.Tensor:
    """Relative-regret loss.

    Args:
        pred_log_t: ``(B, S)`` predicted log runtimes.
        true_log_t: ``(B, S)`` ground-truth log runtimes (for stability).
        true_t: ``(B, S)`` ground-truth runtimes in seconds (linear).
        tau: softmin temperature; smaller -> sharper, harder to optimize.
        clip_regret: cap to keep extreme outliers from dominating.
    """
    if true_t is None:
        true_t = true_log_t.exp()
    # Differentiable argmin via softmin over predicted log-time.
    weights = torch.softmax(-pred_log_t / tau, dim=1)
    expected_t = (weights * true_t).sum(dim=1)
    optimum_t = true_t.min(dim=1).values.clamp_min(1e-9)
    rel_regret = ((expected_t - optimum_t) / optimum_t).clamp_max(clip_regret)
    return rel_regret.mean()


def listwise_loss(
    pred_log_t: torch.Tensor,
    true_log_t: torch.Tensor,
    **_: object,
) -> torch.Tensor:
    """ListMLE: NLL of the true permutation under Plackett-Luce.

    We want fast solvers to score high, so we feed ``-pred_log_t`` as
    scores and order solvers from fastest (smallest true time) to
    slowest. The probability of the observed ordering is
    ``prod_i exp(s_i) / sum_{j>=i} exp(s_j)``.
    """
    scores = -pred_log_t  # higher = predicted faster
    # Sort indices by ground-truth time (ascending = fastest first).
    order = torch.argsort(true_log_t, dim=1)
    sorted_scores = scores.gather(1, order)

    # Cumulative log-sum-exp from the right: for each position i, the
    # normalising constant is logsumexp over positions [i, ..., S-1].
    flipped = sorted_scores.flip(dims=[1])
    cum_lse = torch.logcumsumexp(flipped, dim=1).flip(dims=[1])
    log_prob = (sorted_scores - cum_lse).sum(dim=1)
    return (-log_prob).mean()


LOSS_REGISTRY = {
    "regression": regression_loss,
    "regret": regret_loss,
    "listwise": listwise_loss,
}
