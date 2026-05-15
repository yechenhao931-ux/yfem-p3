"""Ablation studies for the thesis experiments section.

For each ablation, we re-train the model with a subset of input
features zeroed out (or a different hyperparameter) and re-evaluate
mean validation regret. The resulting table directly answers
"which design choices matter?" - a standard request from thesis
reviewers.

Outputs a CSV that can be pasted into LaTeX as a tabular block.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import (
    FEATURE_COLUMNS, SOLVER_NAMES, HeatDataset,
    load_dataset, stratified_group_split,
)
from .evaluate import evaluate
from .losses import LOSS_REGISTRY
from .model import SolverSelector


# Feature groups: which columns to ablate together.
FEATURE_GROUPS: Dict[str, List[str]] = {
    "problem_meta":    ["problem_type", "dim", "order", "ref_levels"],
    "scale":           ["n", "nnz", "avg_nnz", "max_nnz", "density"],
    "spectral_proxy":  ["diag_dom", "cond_est", "frob_norm", "max_abs", "trace"],
    "structure":       ["symmetry", "bandwidth"],
    "physics":         ["kappa", "alpha", "dt", "aniso", "reaction", "velocity"],
}


@dataclass
class AblationResult:
    name: str
    mean_regret: float
    top1: float
    p95_regret: float


def train_one(
    ds: HeatDataset,
    tr_idx: torch.Tensor,
    val_idx: torch.Tensor,
    feature_mask: torch.Tensor,
    args: argparse.Namespace,
) -> AblationResult:
    torch.manual_seed(args.seed)

    x_tr = ds.features[tr_idx] * feature_mask
    x_val = ds.features[val_idx] * feature_mask
    t_tr = ds.times[tr_idx]
    t_val = ds.times[val_idx]
    pt_val = ds.problem_type[val_idx]
    log_t_tr = t_tr.clamp_min(1e-9).log()

    model = SolverSelector(
        num_features=len(FEATURE_COLUMNS),
        num_solvers=len(SOLVER_NAMES),
        hidden=args.hidden,
        depth=args.depth,
        dropout=args.dropout,
    )
    model.set_normalization(
        x_tr.mean(dim=0),
        x_tr.std(dim=0),
        log_t_tr.mean(dim=0),
        log_t_tr.std(dim=0),
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = LOSS_REGISTRY[args.loss]
    loader = DataLoader(
        TensorDataset(x_tr, log_t_tr, t_tr),
        batch_size=args.batch_size, shuffle=True,
    )

    best_score = math.inf
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        for xb, log_tb, tb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, log_tb, true_t=tb, tau=args.regret_tau)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            rep = evaluate(model(x_val), t_val, pt_val)
        if rep.mean_regret < best_score - 1e-6:
            best_score = rep.mean_regret
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        rep = evaluate(model(x_val), t_val, pt_val)
    return AblationResult(
        name="",
        mean_regret=rep.mean_regret,
        top1=rep.top1_accuracy,
        p95_regret=rep.p95_regret,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="Output CSV of ablation results.")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--loss", choices=list(LOSS_REGISTRY.keys()),
                   default="regret")
    p.add_argument("--regret-tau", type=float, default=0.1)
    args = p.parse_args()

    ds = load_dataset(args.csv)
    tr_idx, val_idx = stratified_group_split(
        ds, args.val_frac, args.seed, group_aware=True
    )

    # Baseline: all features on.
    all_on = torch.ones(len(FEATURE_COLUMNS))
    results: List[AblationResult] = []

    base = train_one(ds, tr_idx, val_idx, all_on, args)
    base.name = "full"
    results.append(base)
    print(f"[full]              mean_regret={base.mean_regret:+.4f}  "
          f"top1={base.top1:.3f}")

    # Drop each feature group in turn.
    for group, cols in FEATURE_GROUPS.items():
        mask = all_on.clone()
        for c in cols:
            mask[FEATURE_COLUMNS.index(c)] = 0.0
        res = train_one(ds, tr_idx, val_idx, mask, args)
        res.name = f"drop_{group}"
        results.append(res)
        print(f"[drop {group:<14}] mean_regret={res.mean_regret:+.4f}  "
              f"top1={res.top1:.3f}  "
              f"delta={res.mean_regret - base.mean_regret:+.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ablation", "mean_regret", "top1_accuracy",
                    "p95_regret", "delta_vs_full"])
        for r in results:
            w.writerow([
                r.name, f"{r.mean_regret:.6f}", f"{r.top1:.6f}",
                f"{r.p95_regret:.6f}",
                f"{r.mean_regret - base.mean_regret:+.6f}",
            ])
    print(f"\nWrote ablation results to {args.out}")


if __name__ == "__main__":
    main()
