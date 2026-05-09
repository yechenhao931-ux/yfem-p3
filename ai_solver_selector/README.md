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

- Walks a grid of `(mesh, refinement, order, problem_type, kappa, dt, aniso)`.
- For each instance it assembles either a steady Poisson system (`-Δ u = f`,
  `problem_type=0`) or a transient implicit step `T = M + dt*K` from ex16
  (`problem_type=1`).
- Extracts 17 features (size, nnz, density, diagonal-dominance, symmetry
  defect, condition proxy, problem metadata, etc.).
- Times **6 solver+preconditioner pairs** on the same matrix:
  CG+Jacobi, CG+GS, GMRES+Jacobi, GMRES+GS, BiCGStab+Jacobi, MINRES+Jacobi.
- Writes one CSV row per problem.

Useful flags:

- `-n N` randomly sub-sample N configurations from the full grid (faster
  iteration during development).
- `-rp K` average each solver over K runs (reduces timing jitter).
- `-mi`/`-rt` change the solver's `max_iter` / relative tolerance.

The solver list lives in `solver_registry.hpp`; append-only, so existing
trained models stay valid.

## 2. Train

```bash
cd <repo root>
pip install -r ai_solver_selector/training/requirements.txt
python -m ai_solver_selector.training.train \
    --csv data/heat_solver_dataset.csv \
    --out ai_solver_selector/inference/solver_selector.ts
```

The trainer:

- Splits 85/15 train/val.
- Standardises the features and predicts `log(time)` per solver (runtimes
  span orders of magnitude, so log-space is much better behaved).
- Reports validation loss, top-1 best-solver accuracy, and the average
  *fractional* slowdown of the AI's pick vs. the ground-truth optimum
  (`mean_overhead` 0.05 = picks are ~5% slower on average).
- Saves a TorchScript module (`torch.jit.script`) so C++ can load it without
  Python.

Tune `--epochs`, `--hidden`, `--depth`, `--lr` if you need to.

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
