"""Train the solver selector and export it to TorchScript for C++ inference.

Reads a CSV produced by ``ai_collect_heat_data`` (one row per problem
instance) of the form::

    problem_type,dim,...,aniso,t_solver_0,t_solver_1,...,t_solver_S-1

trains a small MLP that predicts ``log(time)`` for each solver, and writes a
TorchScript module that the C++ inference module can load via libtorch.

Example::

    python -m ai_solver_selector.training.train \
        --csv data/heat_solver_dataset.csv \
        --out ai_solver_selector/inference/solver_selector.ts
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import SolverSelector


# Must match matrix_features.hpp / solver_registry.hpp.
FEATURE_COLUMNS = [
    "problem_type", "dim", "order", "ref_levels", "n",
    "nnz", "avg_nnz", "max_nnz", "density", "diag_dom",
    "symmetry", "frob_norm", "cond_est", "kappa",
    "alpha", "dt", "aniso",
    "bandwidth", "trace", "max_abs", "reaction", "velocity",
]
SOLVER_NAMES = [
    "CG+Jacobi", "CG+GS", "GMRES+Jacobi",
    "GMRES+GS", "BiCGStab+Jacobi", "MINRES+Jacobi",
]


def load_dataset(csv_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    feats: list[list[float]] = []
    times: list[list[float]] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        time_cols = [c for c in fields if c.startswith("t_solver_")]
        if len(time_cols) != len(SOLVER_NAMES):
            raise ValueError(
                f"CSV has {len(time_cols)} solver columns, expected "
                f"{len(SOLVER_NAMES)}"
            )
        missing = [c for c in FEATURE_COLUMNS if c not in fields]
        if missing:
            print(
                f"NOTE: CSV is missing feature columns {missing}; "
                "they will be filled with 0.0 (older collector output)."
            )
        for row in reader:
            try:
                feats.append([float(row.get(c, 0.0) or 0.0) for c in FEATURE_COLUMNS])
                times.append([float(row[c]) for c in time_cols])
            except (ValueError, KeyError):
                continue
    if not feats:
        raise RuntimeError(f"No usable rows in {csv_path}")
    return (
        torch.tensor(feats, dtype=torch.float32),
        torch.tensor(times, dtype=torch.float32),
    )


def split_train_val(
    x: torch.Tensor, y: torch.Tensor, val_frac: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(round(n * val_frac)))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    return x[tr_idx], y[tr_idx], x[val_idx], y[val_idx]


def best_solver_accuracy(pred_log_t: torch.Tensor, true_t: torch.Tensor) -> float:
    pred_best = pred_log_t.argmin(dim=1)
    true_best = true_t.argmin(dim=1)
    return (pred_best == true_best).float().mean().item()


def relative_overhead(pred_log_t: torch.Tensor, true_t: torch.Tensor) -> float:
    """Average ratio (selected / optimal) - 1, in fractional terms.

    Tells us how much slower the AI's pick is compared to the ground-truth
    best. 0.0 means we always nail the optimum; 0.05 means 5% slower on
    average.
    """
    pred_best = pred_log_t.argmin(dim=1)
    selected_t = true_t.gather(1, pred_best.unsqueeze(1)).squeeze(1)
    true_best_t = true_t.min(dim=1).values
    return ((selected_t / true_best_t.clamp_min(1e-12)) - 1.0).mean().item()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="Path to write the TorchScript module.")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    x, t = load_dataset(args.csv)
    log_t = torch.log(t.clamp_min(1e-9))

    x_tr, log_t_tr, x_val, log_t_val = split_train_val(
        x, log_t, args.val_frac, args.seed
    )
    t_tr = torch.exp(log_t_tr)
    t_val = torch.exp(log_t_val)

    feat_mean = x_tr.mean(dim=0)
    feat_std = x_tr.std(dim=0)
    time_log_mean = log_t_tr.mean(dim=0)
    time_log_std = log_t_tr.std(dim=0)

    model = SolverSelector(
        num_features=len(FEATURE_COLUMNS),
        num_solvers=len(SOLVER_NAMES),
        hidden=args.hidden,
        depth=args.depth,
    )
    model.set_normalization(feat_mean, feat_std, time_log_mean, time_log_std)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()

    loader = DataLoader(
        TensorDataset(x_tr, log_t_tr),
        batch_size=args.batch_size, shuffle=True,
    )

    best_val = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(x_val)
            val_loss = loss_fn(pred_val, log_t_val).item()
            acc = best_solver_accuracy(pred_val, t_val)
            overhead = relative_overhead(pred_val, t_val)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:>4} | val_loss={val_loss:.4f} "
                f"| best_solver_acc={acc:.3f} | mean_overhead={overhead:+.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    scripted = torch.jit.script(model)
    scripted.save(str(args.out))
    print(f"Wrote TorchScript model to {args.out}")
    print(f"Solver index -> name:")
    for i, name in enumerate(SOLVER_NAMES):
        print(f"  {i} -> {name}")


if __name__ == "__main__":
    main()
