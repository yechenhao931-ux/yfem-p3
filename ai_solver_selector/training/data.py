"""Dataset loading and group-aware splitting.

The collector emits one CSV row per problem instance. When we naively
shuffle and split into train/val/test, the same ``(mesh, ref_levels)``
configuration can appear in both sets - just with a different kappa or
dt - and the model learns a near-perfect lookup rather than a useful
generalization. This module supports a *group-aware* split, where every
problem instance from the same ``(mesh, ref_levels)`` group lands
entirely in one fold. This is the standard "leave-groups-out" technique
used in autotuning evaluation; the thesis reports both splits so the
gap quantifies leakage.

In addition, ``stratified_group_split`` keeps the proportion of each
``problem_type`` roughly balanced across folds, which matters once we
add non-symmetric problems (convection-diffusion) that are rarer in the
collected grid.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch


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


@dataclass
class HeatDataset:
    features: torch.Tensor          # (N, F)
    times: torch.Tensor             # (N, S)
    problem_type: torch.Tensor      # (N,) int64
    group: torch.Tensor             # (N,) int64, leave-group-out key
    raw_rows: list[dict]            # original CSV rows for traceability


def load_dataset(csv_path: Path) -> HeatDataset:
    feats: list[list[float]] = []
    times: list[list[float]] = []
    raw_rows: list[dict] = []

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        time_cols = [c for c in fields if c.startswith("t_solver_")]
        if len(time_cols) != len(SOLVER_NAMES):
            raise ValueError(
                f"CSV has {len(time_cols)} solver columns; "
                f"expected {len(SOLVER_NAMES)}"
            )
        missing = [c for c in FEATURE_COLUMNS if c not in fields]
        if missing:
            print(
                f"NOTE: CSV is missing feature columns {missing}; "
                "they will be filled with 0.0 (older collector output)."
            )

        # We approximate the (mesh, ref_levels) group by hashing the
        # discrete-valued columns that uniquely identify a mesh family.
        # The collector emits mesh-file name in the log but not the CSV;
        # ``(dim, ref_levels, n / order)`` is a stable surrogate.
        group_keys: dict[tuple, int] = {}

        for row in reader:
            try:
                fvec = [float(row.get(c, 0.0) or 0.0) for c in FEATURE_COLUMNS]
                tvec = [float(row[c]) for c in time_cols]
            except (ValueError, KeyError):
                continue

            key = (
                int(float(row.get("dim", 0.0) or 0.0)),
                int(float(row.get("ref_levels", 0.0) or 0.0)),
                int(float(row.get("order", 0.0) or 0.0)),
                int(float(row.get("n", 0.0) or 0.0)),
            )
            if key not in group_keys:
                group_keys[key] = len(group_keys)

            feats.append(fvec)
            times.append(tvec)
            raw_rows.append({**row, "_group_id": group_keys[key]})

    if not feats:
        raise RuntimeError(f"No usable rows in {csv_path}")

    X = torch.tensor(feats, dtype=torch.float32)
    T = torch.tensor(times, dtype=torch.float32)
    pt = X[:, FEATURE_COLUMNS.index("problem_type")].long()
    group = torch.tensor([r["_group_id"] for r in raw_rows], dtype=torch.long)

    return HeatDataset(features=X, times=T, problem_type=pt,
                       group=group, raw_rows=raw_rows)


def stratified_group_split(
    ds: HeatDataset,
    val_frac: float,
    seed: int,
    group_aware: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (train_idx, val_idx).

    With ``group_aware=True``, complete groups are assigned to val,
    stratified by problem-type. With ``group_aware=False``, individual
    rows are assigned (the easier, leak-prone variant). The thesis
    reports both numbers.
    """
    g = torch.Generator().manual_seed(seed)
    if not group_aware:
        n = ds.features.shape[0]
        perm = torch.randperm(n, generator=g)
        n_val = max(1, int(round(n * val_frac)))
        return perm[n_val:], perm[:n_val]

    # Stratify groups by the majority problem_type within that group.
    rows_by_group: dict[int, list[int]] = defaultdict(list)
    for i, gid in enumerate(ds.group.tolist()):
        rows_by_group[gid].append(i)

    pt_of_group: dict[int, int] = {
        gid: int(ds.problem_type[idxs].mode().values.item())
        for gid, idxs in rows_by_group.items()
    }

    train_idx: list[int] = []
    val_idx: list[int] = []
    groups_by_pt: dict[int, list[int]] = defaultdict(list)
    for gid, pt in pt_of_group.items():
        groups_by_pt[pt].append(gid)

    for pt, gids in groups_by_pt.items():
        order = torch.randperm(len(gids), generator=g).tolist()
        shuffled = [gids[i] for i in order]
        n_val_groups = max(1, int(round(len(shuffled) * val_frac)))
        val_groups = set(shuffled[:n_val_groups])
        for gid in shuffled:
            (val_idx if gid in val_groups else train_idx).extend(rows_by_group[gid])

    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(val_idx, dtype=torch.long),
    )
