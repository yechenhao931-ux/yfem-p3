"""Train the solver selector and export it to TorchScript for C++ inference.

The thesis defends three design choices that this script implements:

1. **Cost-aware objective.**  Instead of generic regression, we minimise
   relative selection regret directly with a softmin-temperature
   relaxation. A listwise (Plackett-Luce) loss is also available.
2. **Group-aware evaluation.**  Train / validation splits are
   stratified by ``(dim, ref_levels, order, n)`` group so the model
   cannot memorise individual mesh configurations.
3. **Comparison against baselines.**  Always-X heuristics, ridge
   regression, and 1-NN are evaluated on the same held-out set so the
   thesis can report the absolute gain from the neural approach.

Example::

    python -m ai_solver_selector.training.train \
        --csv data/heat_solver_dataset.csv \
        --out ai_solver_selector/inference/solver_selector.ts \
        --model moe --loss regret --epochs 600
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import baselines
from .data import (
    FEATURE_COLUMNS, SOLVER_NAMES, HeatDataset,
    load_dataset, stratified_group_split,
)
from .evaluate import EvalReport, evaluate, format_report
from .losses import LOSS_REGISTRY
from .model import MoESolverSelector, SolverSelector


NUM_PROBLEM_TYPES = 5  # 0..4 (see collect_heat_data)


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.model == "mlp":
        return SolverSelector(
            num_features=len(FEATURE_COLUMNS),
            num_solvers=len(SOLVER_NAMES),
            hidden=args.hidden,
            depth=args.depth,
            dropout=args.dropout,
        )
    if args.model == "moe":
        return MoESolverSelector(
            num_features=len(FEATURE_COLUMNS),
            num_solvers=len(SOLVER_NAMES),
            num_problem_types=NUM_PROBLEM_TYPES,
            hidden=args.hidden,
            depth=args.depth,
            dropout=args.dropout,
        )
    raise ValueError(f"unknown model {args.model}")


def cosine_lr(epoch: int, total: int, base_lr: float, warmup: int) -> float:
    if epoch < warmup:
        return base_lr * (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def run_baselines(
    ds: HeatDataset,
    tr_idx: torch.Tensor,
    val_idx: torch.Tensor,
) -> dict[str, EvalReport]:
    x_tr, x_val = ds.features[tr_idx], ds.features[val_idx]
    t_val = ds.times[val_idx]
    pt_val = ds.problem_type[val_idx]
    log_t_tr = ds.times[tr_idx].clamp_min(1e-9).log()

    reports: dict[str, EvalReport] = {}

    # Oracle (lower bound on regret).
    reports["oracle"] = evaluate(
        baselines.oracle(t_val), t_val, pt_val
    )

    # Constant-policy baselines: one per solver.
    for k, name in enumerate(SOLVER_NAMES):
        reports[f"always-{name}"] = evaluate(
            baselines.always_solver(t_val.size(0), t_val.size(1), k),
            t_val, pt_val,
        )

    # Ridge regression per solver, log-time target.
    pred_ridge = baselines.ridge_per_solver(x_tr, log_t_tr, x_val, alpha=1.0)
    reports["ridge"] = evaluate(pred_ridge, t_val, pt_val)

    # 1-NN in standardised feature space.
    pred_nn = baselines.nearest_neighbor(x_tr, log_t_tr, x_val)
    reports["1-nn"] = evaluate(pred_nn, t_val, pt_val)

    return reports


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--metrics-out", type=Path, default=None,
                   help="Optional JSON file to dump metrics for the thesis.")

    # Model.
    p.add_argument("--model", choices=["mlp", "moe"], default="mlp")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)

    # Optimization.
    p.add_argument("--loss", choices=list(LOSS_REGISTRY.keys()),
                   default="regression")
    p.add_argument("--regret-tau", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-epochs", type=int, default=20)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=60,
                   help="Early-stopping patience (0 = disabled).")

    # Data.
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-group-split", action="store_true",
                   help="Disable group-aware splitting (leakage baseline).")

    # Inference-time uncertainty.
    p.add_argument("--mc-samples", type=int, default=0,
                   help="If > 0, also report MC-dropout uncertainty stats.")

    args = p.parse_args()
    torch.manual_seed(args.seed)

    ds = load_dataset(args.csv)
    tr_idx, val_idx = stratified_group_split(
        ds, args.val_frac, args.seed,
        group_aware=not args.no_group_split,
    )
    print(
        f"Loaded {ds.features.size(0)} rows, {len(set(ds.group.tolist()))} "
        f"groups. Split: train={tr_idx.numel()} val={val_idx.numel()} "
        f"(group_aware={not args.no_group_split})."
    )

    x_tr, t_tr = ds.features[tr_idx], ds.times[tr_idx]
    x_val, t_val = ds.features[val_idx], ds.times[val_idx]
    pt_val = ds.problem_type[val_idx]
    log_t_tr = t_tr.clamp_min(1e-9).log()
    log_t_val = t_val.clamp_min(1e-9).log()

    # Per-feature / per-solver statistics from training fold only -
    # critical to avoid validation leakage through normalization.
    feat_mean = x_tr.mean(dim=0)
    feat_std = x_tr.std(dim=0)
    time_log_mean = log_t_tr.mean(dim=0)
    time_log_std = log_t_tr.std(dim=0)

    model = build_model(args)
    model.set_normalization(feat_mean, feat_std, time_log_mean, time_log_std)

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = LOSS_REGISTRY[args.loss]
    loader = DataLoader(
        TensorDataset(x_tr, log_t_tr, t_tr),
        batch_size=args.batch_size, shuffle=True,
    )

    best_score = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    epochs_since_best = 0

    for epoch in range(1, args.epochs + 1):
        lr_now = cosine_lr(epoch - 1, args.epochs, args.lr, args.warmup_epochs)
        for g in opt.param_groups:
            g["lr"] = lr_now

        model.train()
        for xb, log_tb, tb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, log_tb, true_t=tb, tau=args.regret_tau)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(x_val)
            val_report = evaluate(pred_val, t_val, pt_val)
        # Model-selection criterion: mean regret on validation.
        score = val_report.mean_regret
        improved = score < best_score - 1e-6
        if improved:
            best_score = score
            best_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs or improved:
            print(
                f"epoch {epoch:>4} lr={lr_now:.2e} "
                f"| top1={val_report.top1_accuracy:.3f} "
                f"| mean_regret={val_report.mean_regret:+.4f} "
                f"| p95={val_report.p95_regret:+.4f}"
                + ("  *best*" if improved else "")
            )

        if args.patience > 0 and epochs_since_best >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for "
                  f"{args.patience} epochs).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    print("\n=== Final validation report (best checkpoint) ===")
    with torch.no_grad():
        pred_val = model(x_val)
        final_report = evaluate(pred_val, t_val, pt_val)
    print(format_report(final_report, SOLVER_NAMES))

    print("\n=== Baselines on the same validation split ===")
    base_reports = run_baselines(ds, tr_idx, val_idx)
    for name, rep in base_reports.items():
        print(f"\n[{name}]")
        print(format_report(rep, SOLVER_NAMES))

    if args.mc_samples > 0:
        from .model import predict_with_uncertainty
        mean_log_t, std_log_t = predict_with_uncertainty(
            model, x_val, n_samples=args.mc_samples
        )
        max_std = std_log_t.max(dim=1).values
        print(
            f"\nMC-dropout uncertainty: mean(max_std)="
            f"{max_std.mean().item():.4f}, "
            f"p95(max_std)={torch.quantile(max_std, 0.95).item():.4f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model)
    scripted.save(str(args.out))
    print(f"\nWrote TorchScript model to {args.out}")

    if args.metrics_out is not None:
        payload = {
            "config": vars(args) | {"csv": str(args.csv), "out": str(args.out),
                                    "metrics_out": str(args.metrics_out)},
            "neural": _report_to_dict(final_report),
            "baselines": {k: _report_to_dict(v) for k, v in base_reports.items()},
        }
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote metrics to {args.metrics_out}")


def _report_to_dict(r: EvalReport) -> dict:
    return {
        "top1_accuracy": r.top1_accuracy,
        "top3_accuracy": r.top3_accuracy,
        "mean_regret": r.mean_regret,
        "median_regret": r.median_regret,
        "p95_regret": r.p95_regret,
        "fraction_within_5pct": r.fraction_within_5pct,
        "per_problem_type": r.per_problem_type,
        "confusion": r.confusion.tolist(),
    }


if __name__ == "__main__":
    main()
