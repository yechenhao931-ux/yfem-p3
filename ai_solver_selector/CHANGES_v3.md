# AI Solver Selector — v3 修改文档

本次升级聚焦两点：
1. **真实网格数据源**：不再仅依赖 `Mesh::MakeCartesian2D/3D` 生成的规则方/三/六/四面体网格，
   而是支持直接读取 `mfem/data/*.mesh` 中的复杂几何（L 型、星形、Fichera、Beam、混合单元等）。
2. **扩展求解器集合**：从原本的 8 种 (CG/PCG_*/MINRES/MINRES_Jac/GMRES/GMRES_*)
   扩展到 14 种 Krylov 求解器 + 直接法。新增 FGMRES、BiCGSTAB 家族和 `PCG_l1Jac`。

---

## 1. 改动文件

| 文件 | 变更 |
| --- | --- |
| `data_collector/collect_heat_data.cpp` | 真实网格枚举/读取/过滤；扩展求解器列表；新 CLI flag。 |
| `Inference_system/heat_solver_inference.hpp` | `Solve()` 增加 FGMRES / BiCGSTAB / SLI 分支；用统一的"后缀解析"决定预条件子。 |
| `Inference_system/heat_demo.cpp` | 增加 `--mesh-file` 选项，可对单个 `.mesh` 文件做推理 + 求解 + 对比。 |
| `ai_solver_selector/CHANGES_v3.md` | 本文件。 |

`torch_training/train_heat_solver.py` **无需改动**：训练脚本用 `pandas.get_dummies` 动态 one-hot 编码
`mesh_type` 和 `problem_family`，新出现的 `MIXED` / `WEDGE` / `PYRAMID` 会自动展成新列。

---

## 2. 数据生成增强：真实网格

### 2.1 设计

新增 `EnumerateMeshFiles(dir)` + `LoadMeshFromFile(path, ref)`，做了两层过滤：

1. **按文件名过滤**（避免读取明显无法用的文件）：
   - `*nurbs*` (NURBS 网格需要不同的 FEC)
   - `periodic-*` (纯扩散在周期边界上奇异)
   - `klein-*`, `mobius-*`, `escher*`, `*-surf.mesh` (曲面网格)
   - `*segment*` (1D)

2. **按读取后属性过滤**：
   - `Dimension() < 2`
   - `SpaceDimension() != Dimension()` (曲面)
   - `NURBSext != nullptr`

`MakeMesh` 仍用于合成 Cartesian 网格；`LoadMeshFromFile` 用于文件。
两者都返回 `shared_ptr<Mesh>`，统一在主循环里处理。

### 2.2 mesh_type 标签

对于真实网格，`MeshTypeLabel(*mesh)` 检查所有单元的类型：
- 同质 → `QUAD` / `TRI` / `HEX` / `TET` / `WEDGE` / `PYRAMID`
- 混合（如 `square-mixed.mesh`、`fichera-mixed.mesh`）→ `MIXED`

CSV 的 `mesh_type` 列因此会出现新的值（`MIXED`、`WEDGE`、`PYRAMID`）。
Python 训练脚本自动处理：
```python
df_cat = pd.get_dummies(df[["mesh_type", "problem_family"]], prefix=["mesh", "phys"])
```

C++ 推理端 `BuildFeatureVector` 也支持，只是当当前网格类型不在训练集中
时该 one-hot 通道为 0（fallback 行为）。

### 2.3 在 `/home/user/yfem-p3/data` 上实测的入选数量

过滤后保留 **52 个** mesh 文件，涵盖：

```
2D:  amr-quad, beam-quad{,-amr}, beam-tri, channel-bifurcation-2d, compass,
     hexagon, inline-quad, inline-tri, l-shape, mfem, ref-square, ref-triangle,
     rt-2d-{p4-tri,q3}, square-disc{,-p2,-p3}, square-mixed,
     star{,-hilbert,-mixed{,-p2},-q2,-q3}

3D:  amr-hex, beam-{hex,tet,wedge}, equilateral-pyramid, fichera{,-amr,
     -mixed{,-16,-p2},-q2,-q3,-quad{,-mixed}}, inline-{hex,pyramid,tet,wedge},
     llnl-p3, octahedron, ref-{cube,prism,pyramid,tetrahedron},
     square-disc (3D), tinyzoo-3d, toroid-{hex,wedge}
```

NURBS、曲面、1D 网格均已自动跳过。

### 2.4 新 CLI flag

| Flag | 含义 | 默认 |
| --- | --- | --- |
| `--mesh-dir <path>` | 启用从目录读真实网格 | `""` (关闭) |
| `--mesh-min-ref N` | 真实网格最小细化次数 | 0 |
| `--mesh-max-ref N` | 真实网格最大细化次数 | 3 |
| `--mesh-limit N` | 随机抽样 N 个真实网格（再按文件名排序，使输出可复现） | -1 (全部) |
| `--mesh-seed N` | `--mesh-limit` 抽样的随机种子 | 42 |
| `--no-synthetic` | 跳过 Cartesian 合成网格，只用真实网格 | false |
| `--max-dof N` | DOF 超过 N 的 case 直接跳过（防 OOM） | 0 (不限) |

### 2.5 推荐使用方式

```bash
# A) 只用合成网格（向后兼容，与之前完全一致）
./ai_collect_heat_data

# B) 在合成基础上加 mfem/data 真实网格（推荐）
./ai_collect_heat_data \
    --mesh-dir /home/user/yfem-p3/data \
    --mesh-max-ref 3 \
    --max-dof 200000 \
    --output heat_results_v3.csv \
    --output-best heat_best_v3.csv

# C) 只用真实网格 + 少量细化，单跑一遍快速验证
./ai_collect_heat_data \
    --mesh-dir /home/user/yfem-p3/data \
    --no-synthetic \
    --mesh-max-ref 2 \
    --mesh-limit 20 \
    --problem steady \
    --max-dof 80000

# D) 大规模采集 (限制规模避免大网格爆炸)
./ai_collect_heat_data \
    --mesh-dir /home/user/yfem-p3/data \
    --max-ref 5 --max-ref-3d 3 \
    --mesh-max-ref 4 \
    --max-dof 300000 \
    --timeout 60 \
    --output heat_results_big.csv
```

---

## 3. 扩展的求解器集合

| 原有 (8) | 新增 (6) |
| --- | --- |
| `CG` | `PCG_l1Jac` (DSmoother type=1，l1-Jacobi) |
| `PCG_Jacobi` | `FGMRES_Jac` |
| `PCG_GS` | `FGMRES_GS` |
| `MINRES` | `BiCGSTAB` |
| `MINRES_Jac` | `BiCGSTAB_Jac` |
| `GMRES` | `BiCGSTAB_GS` |
| `GMRES_Jac` | |
| `GMRES_GS` | |

加上 `DIRECT_UMF`（条件可用），共 **15 种求解器**。

### 3.1 为什么加这些

- **`PCG_l1Jac`**：l1-Jacobi 在对角占优略差时比标准 Jacobi 更稳健；MFEM 通过 `DSmoother(A, 1)` 即可。
- **`FGMRES` 系列**：在前提条件子本身是迭代型（如复杂的多级方法）时，
  GMRES 的 Krylov 空间假设会被破坏；FGMRES 显式存放前一步预条件后的方向向量，
  允许变前提条件子。对扩散主导但 Jacobi 弱的情况下偶尔比 GMRES 略快。
- **`BiCGSTAB` 系列**：低存储（O(n) 而非 GMRES 的 O(kn)）非对称求解器。
  对中等规模非对称问题（如对流-扩散）非常常用。

### 3.2 命名约定 / 解析规则

`RunIterative` 和 `HeatSolverSelector::Solve` 都按统一规则解析名字：

1. **求解器主类**：按前缀决定
   - `CG` / `PCG_*`        → `CGSolver`
   - `MINRES*`             → `MINRESSolver`
   - `FGMRES*`             → `FGMRESSolver` (必须先于 GMRES 匹配！)
   - `GMRES*`              → `GMRESSolver`
   - `BiCGSTAB*`           → `BiCGSTABSolver`
   - `SLI*`                → `SLISolver`
2. **前提条件子**：取名字第一个 `_` 之后的字段
   - `_GS`        → `GSSmoother`
   - `_Jacobi`/`_Jac` → `DSmoother(A, 0)`
   - `_l1Jac`     → `DSmoother(A, 1)`
   - `_Cheby`     → `DSmoother(A, 2, 10)`  (旧式，保留兼容)
   - 空（如裸 `GMRES`）→ 不用前提条件子

> **重要**：FGMRES 在 `substr(0,5) == "GMRES"` 判断之前匹配 `substr(0,6) == "FGMRES"`，
> 否则 `FGMRES_Jac` 会被当成 GMRES 处理。Solver 端和 Inference 端两处都已修正。

### 3.3 含零对角时的跳过策略

`RunIterative` 中保留原有的 `has_zero_diag` 检查：含零对角的矩阵
会自动跳过所有使用 Jacobi/GS/Chebyshev 的求解器（即 `precond_name != "None"`），
否则 `DSmoother` 会断言失败。`CG` / `MINRES` / `GMRES` / `BiCGSTAB` / `FGMRES`
不需要对角，可以照常运行（虽然 FGMRES 不带预条件子可能不收敛——故 `FGMRES`
没有 `None` 变体，必须配 Jacobi 或 GS）。

---

## 4. 推理端 (`heat_demo.cpp`) 改动

新增 `--mesh-file <path>` 选项：

```bash
# 用模型在真实网格上选最优求解器并实际求解
./heat_demo \
    --model ../torch_training/models/solver_heat_model.json \
    --mesh-file /home/user/yfem-p3/data/star.mesh \
    --ref 2 --order 1 \
    --problem steady

./heat_demo \
    --mesh-file /home/user/yfem-p3/data/fichera.mesh \
    --ref 1 --order 2 \
    --problem aniso --k 50

./heat_demo \
    --mesh-file /home/user/yfem-p3/data/l-shape.mesh \
    --ref 3 \
    --problem multimat --contrast 1000
```

`--mesh-file` 给定时，`--dim` 与 `--mesh` 会被忽略——维度从文件读取后由
`p.mesh->Dimension()` 决定，问题构造里所有 `cfg.dim` 都已改为 `dim_actual`。

NURBS / 曲面网格在此分支会直接报错退出（与数据收集端口径一致，避免静默错误）。

---

## 5. 训练流程的影响

训练脚本本身 **不需要修改**，但新的数据使得：

1. **类别数增加**：从 8 个求解器类升到最多 15 个；
   LabelEncoder 会自动添加新类别。如果某些新求解器的胜场（is_best）很少，可能引发类别极度不平衡——
   `WeightedRandomSampler` 已经处理了平衡。
2. **新 one-hot 列**：`mesh_MIXED`、`mesh_WEDGE`、`mesh_PYRAMID` 会作为新特征列出现。
   网络输入维度 `n_features` 会因此变大；JSON 模型里的 `feature_names` 会反映新列。
   C++ 推理端按特征名查找索引（`feat_index_`），新增的列在合成网格上自动为 0，无需修改。
3. **重新训练**：`solver_heat_model.json` 必须基于新 CSV 重新跑：
   ```bash
   python train_heat_solver.py \
       --data ../data_collector/heat_results_v3.csv \
       --output models/ \
       --epochs 300
   ```

旧的 `solver_heat_model.json` 仍可使用（只是它不知道新 mesh / 新 solver），
所以在数据采集完之前可以继续使用旧模型做基线对比。

---

## 6. 已知约束 / TODO

- **`compass.mesh`** 是 v1.3 格式，可能在旧 MFEM 上读取失败——`LoadMeshFromFile` 用 try/catch 包裹，
  会自动跳过。
- **`*-amr.mesh`**（非共形 AMR 网格）：`UniformRefinement` 在 NC 网格上行为正确，但是
  H1 元的某些操作可能慢；`max_dof` 早退可避免最差情况。
- **真实网格的 ref_level 含义不同**：合成网格 ref=4 大致对应 16×16 / 4×4×4 单元，
  而真实网格 ref=4 是在已有大小基础上再细化 4 次。建议设置 `--mesh-max-ref 3`。
- **未启用的对流-扩散组**：原代码注释了 `// 4. 对流-扩散` 部分（线性 ConvDiff），
  本次没动它；若想让 GMRES/BiCGSTAB/FGMRES 的差异显现得更明显，可解开注释
  + 再跑一次。
- **`PCG_l1Jac`** 仅是 `DSmoother(A, 1)`，并不是真正的代数多重网格；要上 AMG 需要
  并行 MFEM + Hypre，超出本次工作范围。

---

## 7. 完整使用流程

```bash
# 1) 编译数据收集器（与之前一样）
cd /home/user/yfem-p3/ai_solver_selector/data_collector
make

# 2) 用合成网格 + 真实网格收集
./ai_collect_heat_data \
    --mesh-dir /home/user/yfem-p3/data \
    --mesh-max-ref 3 \
    --max-dof 200000 \
    --max-ref 5 \
    --output ../../data/heat_results_v3.csv \
    --output-best ../../data/heat_best_v3.csv \
    --timeout 60

# 3) 训练（脚本不变）
cd ../torch_training
python train_heat_solver.py \
    --data /home/user/yfem-p3/data/heat_results_v3.csv \
    --output models/ \
    --epochs 300

# 4) 编译并跑推理 demo
cd ../Inference_system
make
./heat_demo \
    --model ../torch_training/models/solver_heat_model.json \
    --mesh-file /home/user/yfem-p3/data/star.mesh \
    --ref 2 \
    --problem steady
```

---

## 8. 变更总结一句话

> 用 `mfem/data/*.mesh` 中的真实几何替代/补充原本的 Cartesian 立方体，
> 并把求解器集合从 8 扩到 15（加 FGMRES、BiCGSTAB、l1-Jacobi）；
> CSV / 训练 / 推理三处接口都已对齐，老模型保持向前兼容（仅缺新类别）。
