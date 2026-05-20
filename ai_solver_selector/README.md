# AI Solver Selector for MFEM Heat Conduction

End-to-end pipeline that learns, from MFEM heat-conduction examples (ex1-style
steady Poisson and ex16-style transient implicit step), which iterative solver
+ preconditioner pair will be fastest on a given linear system, and serves
that prediction back to a C++ MFEM application.

```
+-------------------+      +------------------+      +-----------------+
| C++ data collector| ---> | PyTorch training | ---> | C++ inference   |
| (collect_heat_data|      | (train.py +      |      | (libtorch +     |
|  built on MFEM)   |      |  TorchScript)    |      |  MFEM solvers)  |
+-------------------+      +------------------+      +-----------------+
       CSV dataset            solver_selector.ts        argmin -> SolverId
```

## Layout

| Path | Purpose |
| --- | --- |
| `data_collector/matrix_features.hpp` | Feature schema (17 floats) shared by collection and inference. |
| `data_collector/solver_registry.hpp` | The (Krylov solver + preconditioner) pairs the model picks among. |
| `data_collector/collect_heat_data.cpp` | Sweeps mesh / order / kappa / dt / aniso, times every solver, writes CSV. |
| `data_collector/makefile` | Builds `ai_collect_heat_data` against the MFEM build. |
| `training/model.py` | Small MLP (features -> log-time per solver). |
| `training/train.py` | Trains and exports `solver_selector.ts` via `torch.jit.script`. |
| `inference/solver_selector.hpp` | Header-only libtorch wrapper + `MakeBestSolver()` helper. |
| `inference/demo_select.cpp` | Demo: assembles a Poisson system, asks the model, compares against ground truth. |
| `inference/CMakeLists.txt` | Builds the demo against MFEM + libtorch. |

## 1. Collect data

Build MFEM normally first (`make` in the repo root with `MFEM_USE_*`
configured). Then:

```bash
cd ai_solver_selector/data_collector
make
./ai_collect_heat_data -o ../../data/heat_solver_dataset.csv
```

What it does:

- Walks a grid of `(mesh, refinement, order, problem_type, kappa, dt, aniso,
  reaction, velocity)`.
- For each instance it assembles **one of five** heat-physics linear systems:

  | `problem_type` | System | Symmetric? | Notes |
  | --- | --- | --- | --- |
  | 0 | Steady Poisson  `-∇·(κ∇u) = f` (ex1-style) | yes | SPD |
  | 1 | Transient implicit step  `(M + dt·K) u = f` (ex16-style) | yes | SPD; varies dt |
  | 2 | Pure mass solve  `M u = f` | yes | SPD, well-conditioned |
  | 3 | Reaction-diffusion  `(K + r·M) u = f` | yes | SPD; varies r |
  | 4 | Convection-diffusion  `(K + β·∇) u = f` | **no** | CG diverges; GMRES/BiCGStab matter |

- Extracts **22 features** (matrix size, nnz, density, average bandwidth,
  diagonal-dominance, symmetry defect, condition proxy, trace, max-abs,
  Frobenius norm, plus problem metadata: dim, order, ref-levels, kappa,
  alpha, dt, aniso, reaction, velocity).
- Times **6 solver+preconditioner pairs** on the same matrix:
  CG+Jacobi, CG+GS, GMRES+Jacobi, GMRES+GS, BiCGStab+Jacobi, MINRES+Jacobi.
- Writes one CSV row per problem.

Useful flags:

| Flag | Meaning |
| --- | --- |
| `-pt 0,3,4` | Restrict to the listed problem types. |
| `-n N` | Randomly sub-sample N configurations from the full grid. |
| `-rp K` | Average each solver over K runs (reduces timing jitter). |
| `-w / -no-w` | Toggle the untimed warm-up solve (default on). |
| `-q` | Use a much smaller grid for fast iteration. |
| `-a` | Append to an existing CSV instead of overwriting it. |
| `-mx N` | Skip configurations whose DOF count exceeds `N`. |
| `-mi` / `-rt` | Change `max_iter` / relative tolerance. |

The full grid produces several thousand configurations; use `-n` and
`-mx` to keep wall-clock time manageable, or `-q` for a quick sanity run.

The solver list lives in `solver_registry.hpp`; append-only, so existing
trained models stay valid. Older CSVs (with fewer feature columns) are
still trainable - missing columns are filled with zeros.

## 2. Train

```bash
cd <repo root>
pip install -r ai_solver_selector/training/requirements.txt
python -m ai_solver_selector.training.train \
    --csv data/heat_solver_dataset.csv \
    --out ai_solver_selector/inference/solver_selector.ts \
    --model moe --loss regret \
    --metrics-out reports/heat_solver_run.json
```

The trainer:

- **Group-aware split**: train/validation rows from the same
  `(dim, ref_levels, order, n)` group never mix. This prevents the
  model from memorising specific mesh+order combinations and reflects
  real deployment, where a new user problem is unseen at training time.
  `--no-group-split` reverts to the leaky variant for comparison.
- **Predicts `log(time)` per solver**, then picks `argmin` at inference.
  Log-space avoids the regression collapsing onto the largest examples
  (runtimes span 4+ orders of magnitude).
- **Cosine LR schedule with warmup**, AdamW + grad-clipping.
- **Early stopping** on validation mean-regret.
- **Reports**: top-1 / top-3 argmin accuracy, mean / median / p95
  *relative regret* (`(t_picked - t_optimum) / t_optimum`), and the
  same metrics broken down per `problem_type`.
- **Baseline comparison on the same split**: oracle, always-X (one row
  per solver), ridge regression in log-time space, 1-NN.
- Optionally dumps a JSON report (`--metrics-out`) ready to be cited in
  a thesis "Experiments" section.
- Saves a TorchScript module so C++ can load it without Python.

### Method overview (thesis Section X.Y material)

| Component | Choice | Rationale |
| --- | --- | --- |
| Architecture | MLP (`--model mlp`) **or** mixture-of-experts (`--model moe`) | MoE has one head per `problem_type`; the shared trunk learns generic matrix->time relations, heads specialise (e.g. CG vs. GMRES regime). |
| Output target | `log(time)` per solver | Log-space stabilises optimisation when runtimes span orders of magnitude. |
| Loss | Smooth-L1 regression *or* **soft-regret** *or* **ListMLE** (`--loss`) | Regression optimises fit; regret optimises decision quality; listwise optimises ordering. The thesis ablates all three on the same data. |
| Regularisation | Dropout in every hidden block, weight-decay, gradient clipping | Dropout doubles as a Monte Carlo posterior at inference for uncertainty estimates. |
| Schedule | Linear warmup -> cosine decay | Standard recipe; stable for small datasets. |
| Selection criterion | **Validation mean regret**, not validation loss | Loss is a proxy; regret is the deployment objective. |
| Evaluation | Top-1, Top-3 argmin accuracy, mean / median / p95 regret, per-type breakdown, confusion matrix | Top-k tells you whether the model's ranking is roughly right; regret tells you the actual wall-clock cost in seconds; the breakdown shows where the gain comes from. |
| Baselines | Oracle, always-X (six policies), ridge regression, 1-NN | Standard suite in autotuning literature. The neural model has to clear all of them. |

### Soft-regret loss

The classical regression objective treats every (problem, solver) pair
equally; a 10% error on a slow GMRES run gets the same gradient
magnitude as a 10% error on the optimal CG run. For solver selection
only the *argmin* matters, so we minimise *expected relative regret*:

```
y_hat_i = log-time predicted for solver i
w       = softmax(-y_hat / tau)             # soft argmin, differentiable
loss    = E_w[ t_actual ] / min_i t_actual_i - 1
```

As `tau -> 0` the soft argmin becomes the true argmin and the loss
becomes the deterministic regret. At moderate `tau` (default 0.1)
every solver receives gradient signal, which is necessary for stable
training. Set with `--loss regret --regret-tau 0.1`.

### Group-aware splitting

Naive row-shuffled splits leak: the same `(mesh, ref_levels, order)`
configuration appears in train and validation with only `kappa` / `dt`
varying. The model memorises the lookup and reports >95% accuracy that
collapses in production. We split groups, not rows, where a group is
the discrete tuple `(dim, ref_levels, order, n)`. The thesis quantifies
the leakage by training both ways and reporting the gap.

### Monte Carlo dropout uncertainty

Dropout stays enabled at inference for `--mc-samples N` stochastic
forward passes. The per-solver mean is the prediction, the per-solver
standard deviation is an epistemic-uncertainty estimate. The C++
inference module exposes `PredictBestSolverSafe(...)`, which falls
back to a safe default (CG+Jacobi) when uncertainty exceeds a
configurable threshold - the deployment-safety story for the thesis.

### Ablation studies

```bash
python -m ai_solver_selector.training.ablation \
    --csv data/heat_solver_dataset.csv \
    --out reports/ablation.csv
```

Re-trains the model with each feature group masked out and reports the
mean-regret delta vs. the full feature set. The current grouping
follows the matrix-features header:

| Group | Features | What it captures |
| --- | --- | --- |
| `problem_meta` | `problem_type, dim, order, ref_levels` | which PDE we are solving |
| `scale` | `n, nnz, avg_nnz, max_nnz, density` | how big and sparse the matrix is |
| `spectral_proxy` | `diag_dom, cond_est, frob_norm, max_abs, trace` | cheap conditioning surrogates |
| `structure` | `symmetry, bandwidth` | symmetry and locality |
| `physics` | `kappa, alpha, dt, aniso, reaction, velocity` | PDE coefficient regime |

The thesis can cite the resulting table directly.

## 3. Serve from C++

Once you have `solver_selector.ts`:

```cpp
#include "ai_solver_selector/inference/solver_selector.hpp"

ai_ss::SolverSelectorModel model("solver_selector.ts");

mfem::SparseMatrix A; mfem::Vector B;
// ... assemble heat-conduction system into (A, B) ...

auto features = ai_ss::ExtractFeatures(
    /*problem_type=*/0, dim, order, ref_levels,
    kappa, /*alpha=*/0.0, /*dt=*/0.0, aniso, A);

int picked = -1;
auto sp = ai_ss::MakeBestSolver(model, features, A,
                                /*max_iter=*/2000, /*rtol=*/1e-10, &picked);

mfem::Vector x(B.Size()); x = 0.0;
sp.solver->Mult(B, x);
std::cout << "AI picked " << ai_ss::SolverName(picked) << "\n";
```

Build with libtorch:

```bash
cd ai_solver_selector/inference
cmake -B build \
      -DTorch_DIR=/path/to/libtorch/share/cmake/Torch \
      -DMFEM_DIR=$(pwd)/../../build
cmake --build build
./build/ai_solver_demo -m solver_selector.ts -mesh data/star.mesh -r 4 -o 2
```

The demo prints the model's predicted runtime per solver, runs the chosen
one, and then re-runs all six to confirm whether the AI matched the true
optimum.

## Using real meshes from `mfem/data`

The data collector can now pull `.mesh` files from a directory in addition to
generating Cartesian grids. See `CHANGES_v3.md` for the full design. Quick
recipe:

```bash
./ai_collect_heat_data \
    --mesh-dir /home/user/yfem-p3/data \
    --mesh-max-ref 3 \
    --max-dof 200000 \
    --output ../../data/heat_results_v3.csv
```

NURBS / periodic / surface / 1-D meshes are automatically filtered out. Mixed
meshes (e.g. `square-mixed.mesh`, `fichera-mixed.mesh`) are tagged
`mesh_type=MIXED` and `pandas.get_dummies` handles the new one-hot column
without any change to the trainer.

The serial solver registry now contains 14 Krylov pairs (CG, PCG_Jacobi,
PCG_l1Jac, PCG_GS, MINRES, MINRES_Jac, GMRES, GMRES_Jac, GMRES_GS, FGMRES_Jac,
FGMRES_GS, BiCGSTAB, BiCGSTAB_Jac, BiCGSTAB_GS) plus the optional UMFPACK
direct solver. The inference module (`heat_solver_inference.hpp::Solve`) knows
how to instantiate each one — names are parsed by `prefix_solver_suffix`, so
adding `FGMRES_l1Jac` later only needs an entry in `GetSolverList`.

## Adding new heat-conduction problem types or solvers

1. Append a new entry to `enum SolverId` and `MakeSolver` in
   `solver_registry.hpp`. Existing models still work; new column appears as
   `t_solver_<N>` in fresh CSVs.
2. To add features, append to `enum FeatureIndex` and `kFeatureNames` in
   `matrix_features.hpp`, and update `FillMatrixFeatures` / `ExtractFeatures`.
3. Re-run the collector, retrain, redeploy the new `solver_selector.ts`.

## Notes

- Timings are wall-clock, single-threaded, on the host that runs the
  collector. Re-collect when you move the model to a meaningfully different
  CPU; runtimes are not portable.
- The collector deliberately penalises non-converging solvers so the model
  never picks them.
- For parallel (HypreParMatrix / `ex16p`) workloads, the same recipe works:
  swap the solver registry to `HyprePCG` / `HypreBoomerAMG` etc. and replace
  the `SparseMatrix` feature extractor with one that walks the local CSR.
