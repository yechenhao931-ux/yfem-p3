"""Evaluation metrics for solver selection.

The headline number in the thesis is *mean relative regret*

    regret(i) = (t[i, picked(i)] - min_s t[i, s]) / min_s t[i, s]

which directly measures the wall-clock cost of consulting the AI vs.
running the oracle's best solver. We report mean, median, and 95th
percentile, plus the more standard Top-k argmin accuracy. Per-problem-
type breakdown is included so the thesis can discuss where the model
helps most.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch


@dataclass
class EvalReport:
    top1_accuracy: float
    top3_accuracy: float
    mean_regret: float
    median_regret: float
    p95_regret: float
    fraction_within_5pct: float
    per_problem_type: Dict[int, Dict[str, float]]
    confusion: torch.Tensor  # (S, S) int64: rows=true best, cols=picked


def _regret(picks: torch.Tensor, true_t: torch.Tensor) -> torch.Tensor:
    chosen = true_t.gather(1, picks.unsqueeze(1)).squeeze(1)
    optimum = true_t.min(dim=1).values.clamp_min(1e-12)
    return (chosen - optimum) / optimum


def evaluate(
    pred_log_t: torch.Tensor,
    true_t: torch.Tensor,
    problem_type: torch.Tensor,
    k: int = 3,
) -> EvalReport:
    num_solvers = true_t.size(1)
    picks = pred_log_t.argmin(dim=1)
    true_best = true_t.argmin(dim=1)

    # Top-k: is the actual best among the model's k smallest predictions?
    topk_idx = pred_log_t.topk(k, dim=1, largest=False).indices
    in_topk = (topk_idx == true_best.unsqueeze(1)).any(dim=1).float()

    top1 = (picks == true_best).float().mean().item()
    topk = in_topk.mean().item()

    regret = _regret(picks, true_t)
    mean_regret = regret.mean().item()
    median_regret = regret.median().item()
    p95_regret = torch.quantile(regret, 0.95).item()
    within_5 = (regret <= 0.05).float().mean().item()

    per_pt: Dict[int, Dict[str, float]] = {}
    for pt in problem_type.unique().tolist():
        mask = problem_type == pt
        if mask.sum() == 0:
            continue
        pt_picks = picks[mask]
        pt_best = true_best[mask]
        pt_regret = regret[mask]
        per_pt[int(pt)] = {
            "n": int(mask.sum().item()),
            "top1": (pt_picks == pt_best).float().mean().item(),
            "mean_regret": pt_regret.mean().item(),
            "p95_regret": torch.quantile(pt_regret, 0.95).item(),
        }

    conf = torch.zeros((num_solvers, num_solvers), dtype=torch.long)
    for tb, pk in zip(true_best.tolist(), picks.tolist()):
        conf[tb, pk] += 1

    return EvalReport(
        top1_accuracy=top1,
        top3_accuracy=topk,
        mean_regret=mean_regret,
        median_regret=median_regret,
        p95_regret=p95_regret,
        fraction_within_5pct=within_5,
        per_problem_type=per_pt,
        confusion=conf,
    )


def format_report(report: EvalReport, solver_names: Iterable[str]) -> str:
    lines = [
        f"  top1_accuracy         = {report.top1_accuracy:.3f}",
        f"  top3_accuracy         = {report.top3_accuracy:.3f}",
        f"  mean_regret           = {report.mean_regret:+.4f}",
        f"  median_regret         = {report.median_regret:+.4f}",
        f"  p95_regret            = {report.p95_regret:+.4f}",
        f"  fraction_within_5pct  = {report.fraction_within_5pct:.3f}",
    ]
    if report.per_problem_type:
        lines.append("  per problem_type:")
        for pt, m in sorted(report.per_problem_type.items()):
            lines.append(
                f"    pt={pt} n={m['n']:<4d} top1={m['top1']:.3f}  "
                f"mean_regret={m['mean_regret']:+.4f}  "
                f"p95_regret={m['p95_regret']:+.4f}"
            )
    names = list(solver_names)
    if report.confusion.numel() > 0:
        lines.append("  confusion (rows = true best, cols = picked):")
        header = " " * 8 + " ".join(f"{n[:10]:>10}" for n in names)
        lines.append(header)
        for i, row in enumerate(report.confusion.tolist()):
            lines.append(
                f"    {names[i][:6]:>6}  " + " ".join(f"{c:>10d}" for c in row)
            )
    return "\n".join(lines)
