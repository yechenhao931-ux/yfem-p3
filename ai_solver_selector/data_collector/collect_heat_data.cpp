/**
 * heat_benchmark.cpp  (v2, robust)
 * ================================
 * 基于 MFEM 的热传导问题最优求解器数据收集程序。
 *
 * v2 修复：
 *   - 三角/四面体网格初始化改用 2x2 / 2x2x2 起步，避免 UniformRefinement 退化单元
 *   - BilinearForm::Finalize(1)  保证排序 CSR，满足 EliminateRowCol 要求
 *   - Jacobi/Chebyshev smoother 会检测零对角 → 发现即跳过该求解器
 *   - 非线性问题用解析温度场（x+y+z）替代 Randomize，保证场平滑
 *   - 瞬态 dt 权重通过 lambda+coefficient 叠加，避免 DomainIntegrator 生命周期问题
 *   - 所有求解器调用包一层 try/catch（MFEM 异常 + std::exception）
 *   - 初始 x 向量总是清零
 *   - 对 GS/Chebyshev 的前提（对角必须非零）做前置检查
 *
 * 覆盖的物理模型：
 *   1. 稳态热传导        —  −∇·(k∇T) = Q
 *   2. 各向异性热传导     —  −∇·(K∇T) = Q,  K 为对角各向异性张量
 *   3. 非均匀介质热传导    —  −∇·(k(x)∇T) = Q,  k(x) 含夹杂物跳跃
 *   4. 对流-扩散热传导    —  −∇·(k∇T) + v·∇T = Q  （带 Péclet 数控制）
 *   5. 带辐射边界的热传导  —  Robin 边界 k∂T/∂n + h T = h T∞
 *   6. 瞬态热传导（隐式）  —  (M/∆t + K) T^{n+1} = ...
 *   7. 非线性热传导       —  −∇·(k(T)∇T) = Q,  k 依赖温度（Newton 一步）
 *
 * 覆盖的求解器：
 *   - CG / PCG(GS) / PCG(Jacobi) / PCG(Chebyshev)
 *   - MINRES / MINRES(Jacobi)
 *   - GMRES / GMRES(Jacobi) / GMRES(GS)
 *   - DIRECT_UMFPACK（可选）
 *
 * 每个 (物理, 网格, 规模, 求解器) 组合记录：
 *   - 问题特征（DOF、nnz、对称性、条件数估计）
 *   - 组装 / 配置 / 求解时间
 *   - 迭代次数、收敛状态、残差
 *   - 最优求解器标签（同一问题规模下耗时最短）
 *
 * 输出：
 *   - heat_results.csv   （详细记录，所有求解器）
 *   - heat_best.csv      （每组最优求解器，可直接用于 ML 训练）
 *
 * 编译（示例）：
 *   g++ -std=c++17 -O3 -march=native \
 *       -I$(MFEM_DIR)/include \
 *       heat_benchmark.cpp \
 *       -L$(MFEM_DIR)/lib -lmfem -lm \
 *       -o heat_benchmark
 *   # 可选：-DMFEM_USE_SUITESPARSE -lumfpack -lamd
 *
 * 运行：
 *   ./heat_benchmark                             # 全量默认
 *   ./heat_benchmark --max-ref 4                 # 限制精炼次数
 *   ./heat_benchmark --problem aniso             # 仅各向异性
 *   ./heat_benchmark --output heat_data.csv      # 自定义输出
 *   ./heat_benchmark --dim 3 --order 2           # 3D P2 元
 *   ./heat_benchmark --no-direct --timeout 15    # 跳过直接法，15s 超时
 */

#include "mfem.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace mfem;

// =====================================================================
//  配置
// =====================================================================
struct Config {
    int         min_ref_2d    = 1;
    int         max_ref_2d    = 6;
    int         min_ref_3d    = 1;
    int         max_ref_3d    = 4;
    int         order         = 1;
    double      rtol          = 1e-8;
    double      atol          = 1e-14;
    int         max_iter      = 5000;
    int         gmres_kdim    = 30;

    // 过滤开关
    bool run_steady     = true;
    bool run_aniso      = true;
    bool run_multi_mat  = true;
    bool run_convdiff   = true;
    bool run_robin      = true;
    bool run_transient  = true;
    bool run_nonlinear  = true;

    bool run_2d = true;
    bool run_3d = true;

    // 网格类型开关（用于排查崩溃）
    bool run_quad = true;
    bool run_tri  = true;
    bool run_hex  = true;
    bool run_tet  = true;

    // 直接法 DOF 上限
    int  direct_max_dof = 80000;
    bool skip_direct    = false;

    // 求解器超时（秒）
    double solver_timeout_s = 30.0;

    std::string output_all  = "heat_results.csv";
    std::string output_best = "heat_best.csv";

    bool verbose = false;
};

Config ParseArgs(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--max-ref"    && i+1<argc) c.max_ref_2d = std::stoi(argv[++i]);
        else if (a == "--max-ref-3d" && i+1<argc) c.max_ref_3d = std::stoi(argv[++i]);
        else if (a == "--min-ref"    && i+1<argc) c.min_ref_2d = std::stoi(argv[++i]);
        else if (a == "--order"      && i+1<argc) c.order      = std::stoi(argv[++i]);
        else if (a == "--rtol"       && i+1<argc) c.rtol       = std::stod(argv[++i]);
        else if (a == "--max-iter"   && i+1<argc) c.max_iter   = std::stoi(argv[++i]);
        else if (a == "--timeout"    && i+1<argc) c.solver_timeout_s = std::stod(argv[++i]);
        else if (a == "--direct-max" && i+1<argc) c.direct_max_dof   = std::stoi(argv[++i]);
        else if (a == "--output"     && i+1<argc) c.output_all  = argv[++i];
        else if (a == "--output-best"&& i+1<argc) c.output_best = argv[++i];
        else if (a == "--dim"        && i+1<argc) {
            std::string d = argv[++i];
            if (d == "2") c.run_3d = false;
            if (d == "3") c.run_2d = false;
        }
        else if (a == "--mesh"       && i+1<argc) {
            std::string m = argv[++i];
            c.run_quad = c.run_tri = c.run_hex = c.run_tet = false;
            if (m == "quad" || m == "all") c.run_quad = true;
            if (m == "tri"  || m == "all") c.run_tri  = true;
            if (m == "hex"  || m == "all") c.run_hex  = true;
            if (m == "tet"  || m == "all") c.run_tet  = true;
        }
        else if (a == "--problem"    && i+1<argc) {
            std::string p = argv[++i];
            c.run_steady = c.run_aniso = c.run_multi_mat = false;
            c.run_convdiff = c.run_robin = c.run_transient = c.run_nonlinear = false;
            if (p == "steady"    || p == "all") c.run_steady    = true;
            if (p == "aniso"     || p == "all") c.run_aniso     = true;
            if (p == "multimat"  || p == "all") c.run_multi_mat = true;
            if (p == "convdiff"  || p == "all") c.run_convdiff  = true;
            if (p == "robin"     || p == "all") c.run_robin     = true;
            if (p == "transient" || p == "all") c.run_transient = true;
            if (p == "nonlinear" || p == "all") c.run_nonlinear = true;
        }
        else if (a == "--no-direct") c.skip_direct = true;
        else if (a == "--no-tri")    c.run_tri = false;
        else if (a == "--no-tet")    c.run_tet = false;
        else if (a == "--verbose")   c.verbose = true;
    }
    return c;
}

// =====================================================================
//  结果结构
// =====================================================================
struct HeatResult {
    std::string problem;
    std::string mesh_type;
    int         dim;
    int         ref_level;
    int         poly_order;
    int         n_elements;
 
    int    n_dof;
    long   nnz;
    double nnz_per_row;
    double asymmetry_rel;    // 相对非对称度 (0=完全对称)
    double asymmetry_abs;    // 绝对非对称度 max|A_ij - A_ji|
    double matrix_norm_inf;  // 矩阵行和最大值（归一化参考）
    bool   is_spd;
    double diag_dominance;
    double cond_estimate;
 
    double k_ratio;
    double peclet;
    double material_contrast;
 
    std::string solver_name;
    std::string precond_name;
 
    double assemble_ms;
    double setup_ms;
    double solve_ms;
    double total_ms;
    bool   converged;
    int    iterations;
    double final_residual;
    double rel_residual;
 
    bool is_best = false;
};

// =====================================================================
//  计时
// =====================================================================
using Clock  = std::chrono::high_resolution_clock;
using Millis = std::chrono::duration<double, std::milli>;
inline double since(Clock::time_point t) { return Millis(Clock::now() - t).count(); }

// =====================================================================
//  矩阵特征
// =====================================================================
struct MatStats {
    long   nnz;
    double nnz_per_row;
    double asymmetry_rel;    // 相对非对称度（0=完全对称，1=完全反对称）
    double asymmetry_abs;    // 绝对非对称度 max|A[i,j]-A[j,i]|
    double matrix_norm_inf;  // 矩阵无穷范数（用于归一化判断）
    bool   is_spd;
    double diag_dominance;
    double cond_estimate;
    bool   has_zero_diag;
};
 
    // 计算 A[i,j]：CSR 格式线性扫描对应行（假设未排序也能正确工作）
static double CSR_Get(const SparseMatrix& A, int row, int col) {
    const int*    I = A.GetI();
    const int*    J = A.GetJ();
    const double* V = A.GetData();
    for (int k = I[row]; k < I[row + 1]; ++k) {
        if (J[k] == col) return V[k];
    }
    return 0.0;
}
MatStats ComputeMatStats(const SparseMatrix& A) {
    MatStats s{};
    int N = A.Height();
    s.nnz           = A.NumNonZeroElems();
    s.nnz_per_row   = (double)s.nnz / std::max(N, 1);
    s.has_zero_diag = false;
 
    const int*    I = A.GetI();
    const int*    J = A.GetJ();
    const double* V = A.GetData();
 
    // ── 对称性（全扫描，避免抽样偏差） ─────────────────────
    // 定义：
    //   asymmetry_rel = Σ|a_ij - a_ji| / Σ(|a_ij| + |a_ji|)
    //     0 = 完全对称，1 = 完全反对称（a_ij = -a_ji）
    //   asymmetry_abs = max_{i≠j} |a_ij - a_ji|
    //     可用于区分"相对小但绝对值大"的情况
    //   matrix_norm_inf = max_i Σ_j|a_ij|   提供归一化参考
    double asym_acc = 0.0, mag_acc = 0.0;
    double asym_max = 0.0;
    double row_sum_max = 0.0;
    for (int row = 0; row < N; ++row) {
        double row_abs_sum = 0.0;
        for (int k = I[row]; k < I[row + 1]; ++k) {
            double v = V[k];
            row_abs_sum += std::abs(v);
            int col = J[k];
            if (col == row) continue;
            double aji   = CSR_Get(A, col, row);
            double diff  = std::abs(v - aji);
            double denom = std::abs(v) + std::abs(aji);
            asym_acc += diff;
            mag_acc  += denom;
            if (diff > asym_max) asym_max = diff;
        }
        if (row_abs_sum > row_sum_max) row_sum_max = row_abs_sum;
    }
    s.asymmetry_rel   = (mag_acc > 1e-30) ? asym_acc / mag_acc : 0.0;
    s.asymmetry_abs   = asym_max;
    s.matrix_norm_inf = row_sum_max;
 
    // ── SPD + 对角优势 + 零对角 ─────────────────────────────
    int spd_cnt = 0;
    double dd_acc = 0.0;
    double diag_max = 0.0, diag_min = 1e30;
    const double ZERO_DIAG_TOL = 1e-14;
    for (int row = 0; row < N; ++row) {
        double diag = 0.0, off = 0.0;
        bool found_diag = false;
        for (int k = I[row]; k < I[row+1]; ++k) {
            if (J[k] == row) { diag = V[k]; found_diag = true; }
            else             { off += std::abs(V[k]); }
        }
        if (!found_diag || std::abs(diag) < ZERO_DIAG_TOL)
            s.has_zero_diag = true;
        if (diag > off) ++spd_cnt;
        double dd = (off > 1e-30) ? diag / off : 10.0;
        dd_acc  += std::min(dd, 10.0);
        double d = std::abs(diag);
        if (d > 1e-30) {
            diag_max = std::max(diag_max, d);
            diag_min = std::min(diag_min, d);
        }
    }
    // 判定 SPD：Gershgorin 通过 + 相对对称度好
    s.is_spd         = (spd_cnt == N) && (s.asymmetry_rel < 0.01);
    s.diag_dominance = dd_acc / std::max(N, 1);
 
    // ── 条件数粗估 ─────────────────────────────────────────
    Vector v(N), w(N);
    v.Randomize(42);
    double lam_max = 1.0;
    for (int it = 0; it < 25; ++it) {
        double nv = v.Norml2();
        if (nv < 1e-30) break;
        v /= nv;
        A.Mult(v, w);
        lam_max = w * v;
        v = w;
    }
    lam_max = std::abs(lam_max);
    double cond = (diag_min > 1e-30) ? lam_max / diag_min : 1e6;
    s.cond_estimate = std::log10(std::max(1.0, cond));
    return s;
}


// =====================================================================
//  自定义系数
// =====================================================================
class AnisoK : public MatrixCoefficient {
    double kx_, ky_, kz_;
public:
    AnisoK(int dim, double kx, double ky, double kz = 1.0)
        : MatrixCoefficient(dim), kx_(kx), ky_(ky), kz_(kz) {}
    void Eval(DenseMatrix& K, ElementTransformation& T,
              const IntegrationPoint& ip) override {
        int d = GetHeight();
        K.SetSize(d); K = 0.0;
        K(0,0) = kx_;
        if (d > 1) K(1,1) = ky_;
        if (d > 2) K(2,2) = kz_;
    }
};

class HeterogeneousK : public Coefficient {
    double k_bg_, k_inc_;
    Vector center_;
    double radius_;
public:
    HeterogeneousK(double k_bg, double k_inc, Vector c, double r)
        : k_bg_(k_bg), k_inc_(k_inc), center_(std::move(c)), radius_(r) {}
    double Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
        Vector x(center_.Size()); T.Transform(ip, x);
        double d2 = 0.0;
        for (int i = 0; i < x.Size(); ++i)
            d2 += (x(i) - center_(i)) * (x(i) - center_(i));
        return (std::sqrt(d2) < radius_) ? k_inc_ : k_bg_;
    }
};

// 温度依赖导热 — 用解析表达式替代 GridFunction::Randomize
// k(x) = k0 * (1 + alpha * (x + y + z) / dim)
class SmoothTempDepK : public Coefficient {
    double k0_, alpha_;
public:
    SmoothTempDepK(double k0, double alpha) : k0_(k0), alpha_(alpha) {}
    double Eval(ElementTransformation& T, const IntegrationPoint& ip) override {
        int dim = T.GetDimension();
        Vector x(dim); T.Transform(ip, x);
        double sum = 0.0;
        for (int i = 0; i < dim; ++i) sum += x(i);
        double T_prev = sum / dim;     // 0–1 之间的"温度场"
        return k0_ * (1.0 + alpha_ * T_prev);
    }
};

// =====================================================================
//  求解器列表
// =====================================================================
struct SolverSpec {
    std::string name;
    std::string precond;
    bool        supports_nonsym;
    bool        is_direct;
    bool        needs_diag;      // Jacobi/Chebyshev 需要非零对角
};

std::vector<SolverSpec> GetSolverList(bool has_direct) {
    std::vector<SolverSpec> list = {
        {"CG",          "None",      false, false, false},
        {"PCG_GS",      "GS",        false, false, true },
        {"PCG_Jacobi",  "Jacobi",    false, false, true },
        //{"PCG_Cheby",   "Chebyshev", false, false, true },
        {"MINRES",      "None",      false, false, false},
        {"MINRES_Jac",  "Jacobi",    false, false, true },
        {"GMRES",       "None",      true,  false, false},
        {"GMRES_Jac",   "Jacobi",    true,  false, true },
        {"GMRES_GS",    "GS",        true,  false, true },
    };
    if (has_direct) {
        list.push_back({"DIRECT_UMF", "LU", true, true, false});
    }
    return list;
}

// =====================================================================
//  求解器运行（带异常保护）
// =====================================================================
struct SolveOutput {
    bool   converged;
    int    iters;
    double final_res;
    double solve_ms;
    double setup_ms;
    bool   skipped;             // v2: true 表示跳过（先决条件不满足）
};

SolveOutput RunIterative(const std::string& solver_name,
                          const std::string& precond_name,
                          SparseMatrix& A,
                          const Vector& B, Vector& x,
                          const Config& cfg,
                          bool has_zero_diag)
{
    SolveOutput out{false, 0, 0.0, 0.0, 0.0, false};

    // v2: 若矩阵含零对角，跳过 Jacobi/Chebyshev/GS（会触发 MFEM 断言）
    if (has_zero_diag && precond_name != "None") {
        out.skipped = true;
        return out;
    }

    std::unique_ptr<Solver> prec;

    try {
        auto t_setup = Clock::now();
        if      (precond_name == "GS")        prec.reset(new GSSmoother(A));
        else if (precond_name == "Jacobi")    prec.reset(new DSmoother(A, 0));
        else if (precond_name == "Chebyshev") prec.reset(new DSmoother(A, 2, 10));
        out.setup_ms = since(t_setup);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "    [warn] %s setup failed: %s\n",
                     precond_name.c_str(), e.what());
        out.skipped = true;
        return out;
    }

    std::unique_ptr<IterativeSolver> solver_owner;
    IterativeSolver* solver = nullptr;

    if (solver_name == "CG" || solver_name.substr(0, 3) == "PCG") {
        auto* cg = new CGSolver();
        solver_owner.reset(cg); solver = cg;
    } else if (solver_name.substr(0, 6) == "MINRES") {
        auto* mr = new MINRESSolver();
        solver_owner.reset(mr); solver = mr;
    } else if (solver_name.substr(0, 5) == "GMRES") {
        auto* gm = new GMRESSolver();
        gm->SetKDim(cfg.gmres_kdim);
        solver_owner.reset(gm); solver = gm;
    } else if (solver_name == "BiCGSTAB" || solver_name == "BiCGSTAB_Jacobi") {
      auto* bs = new BiCGSTABSolver();
      solver_owner.reset(bs);
      solver = bs;
   }else {
        out.skipped = true;
        return out;
    }

    try {
        solver->SetOperator(A);
        if (prec) solver->SetPreconditioner(*prec);
        solver->SetRelTol(cfg.rtol);
        solver->SetAbsTol(cfg.atol);
        solver->SetMaxIter(cfg.max_iter);
        solver->SetPrintLevel(cfg.verbose ? 1 : -1);

        x = 0.0;
        auto t_solve = Clock::now();
        solver->Mult(B, x);
        out.solve_ms = since(t_solve);

        out.converged = solver->GetConverged();
        out.iters     = solver->GetNumIterations();
        out.final_res = solver->GetFinalNorm();

        if (!std::isfinite(out.final_res)) {
            out.converged = false;
            out.final_res = 1e30;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "    [warn] %s solve failed: %s\n",
                     solver_name.c_str(), e.what());
        out.converged = false;
    }

    return out;
}

#ifdef MFEM_USE_SUITESPARSE
SolveOutput RunDirect(SparseMatrix& A, const Vector& B, Vector& x)
{
    SolveOutput out{false, 1, 0.0, 0.0, 0.0, false};
    try {
        auto t_setup = Clock::now();
        UMFPackSolver direct;
        direct.SetOperator(A);
        out.setup_ms = since(t_setup);

        x = 0.0;
        auto t_solve = Clock::now();
        direct.Mult(B, x);
        out.solve_ms = since(t_solve);
        out.converged = true;
        out.final_res = 0.0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "    [warn] UMFPack failed: %s\n", e.what());
        out.converged = false;
    }
    return out;
}
#endif

// =====================================================================
//  组装系统封装
// =====================================================================
struct AssembledSystem {
    std::shared_ptr<BilinearForm>       form;
    std::shared_ptr<LinearForm>         lf;
    std::shared_ptr<GridFunction>       gf;
    std::shared_ptr<Coefficient>        k_coef;
    std::shared_ptr<Coefficient>        inv_dt_coef;
    std::shared_ptr<MatrixCoefficient>  K_coef;
    std::shared_ptr<VectorCoefficient>  v_coef;
    std::shared_ptr<Coefficient>        src_coef;
    std::shared_ptr<Coefficient>        robin_coef;
    OperatorPtr A;
    Vector      B, X;
    double      assemble_ms = 0.0;
    bool        valid       = true;
};

// v2: 统一的 Form + BC + Finalize 封装
// finalize_flag = 1 → sorted CSR，EliminateRowCol 需要
void AssembleAndFormSystem(BilinearForm& a, LinearForm& b,
                            FiniteElementSpace& fes, Mesh& mesh,
                            OperatorPtr& A, Vector& X, Vector& B,
                            GridFunction& x_gf,
                            bool apply_dirichlet_all = true,
                            double dirichlet_val = 0.0)
{
    a.Assemble(0);
    a.Finalize(1);          // v2: skip_zeros=1（排序 CSR，支持消元）

    b.Assemble();

    Array<int> ess_tdof_list;
    if (apply_dirichlet_all && mesh.bdr_attributes.Size() > 0) {
        Array<int> ess_bdr(mesh.bdr_attributes.Max());
        ess_bdr = 1;
        fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }
    x_gf = dirichlet_val;
    a.FormLinearSystem(ess_tdof_list, x_gf, b, A, X, B);
}

// ── 1. 稳态均匀 ─────────────────────────────────────────────
AssembledSystem BuildSteadyHeat(Mesh& mesh, FiniteElementSpace& fes,
                                 double k_val, HeatResult& r)
{
    AssembledSystem sys;
    auto t0 = Clock::now();

    sys.form.reset(new BilinearForm(&fes));
    sys.k_coef.reset(new ConstantCoefficient(k_val));
    sys.form->AddDomainIntegrator(new DiffusionIntegrator(*sys.k_coef));

    sys.lf.reset(new LinearForm(&fes));
    sys.src_coef.reset(new ConstantCoefficient(1.0));
    sys.lf->AddDomainIntegrator(new DomainLFIntegrator(*sys.src_coef));

    sys.gf.reset(new GridFunction(&fes));
    AssembleAndFormSystem(*sys.form, *sys.lf, fes, mesh,
                          sys.A, sys.X, sys.B, *sys.gf);
    sys.assemble_ms = since(t0);

    r.k_ratio = 1.0; r.peclet = 0.0; r.material_contrast = 1.0;
    return sys;
}

// ── 2. 各向异性 ─────────────────────────────────────────────
AssembledSystem BuildAnisoHeat(Mesh& mesh, FiniteElementSpace& fes,
                                double kx, double ky, double kz,
                                HeatResult& r)
{
    AssembledSystem sys;
    auto t0 = Clock::now();

    sys.form.reset(new BilinearForm(&fes));
    sys.K_coef.reset(new AnisoK(mesh.Dimension(), kx, ky, kz));
    sys.form->AddDomainIntegrator(new DiffusionIntegrator(*sys.K_coef));

    sys.lf.reset(new LinearForm(&fes));
    sys.src_coef.reset(new ConstantCoefficient(1.0));
    sys.lf->AddDomainIntegrator(new DomainLFIntegrator(*sys.src_coef));

    sys.gf.reset(new GridFunction(&fes));
    AssembleAndFormSystem(*sys.form, *sys.lf, fes, mesh,
                          sys.A, sys.X, sys.B, *sys.gf);
    sys.assemble_ms = since(t0);

    double kmax = std::max({kx, ky, kz});
    double kmin = std::min({kx, ky, kz});
    r.k_ratio = kmax / kmin; r.peclet = 0.0; r.material_contrast = 1.0;
    return sys;
}

// ── 3. 多材料（圆/球形夹杂物）──────────────────────────────
AssembledSystem BuildMultiMatHeat(Mesh& mesh, FiniteElementSpace& fes,
                                   double contrast, HeatResult& r)
{
    AssembledSystem sys;
    auto t0 = Clock::now();

    int dim = mesh.Dimension();
    Vector center(dim); center = 0.5;
    double radius = 0.25;

    sys.form.reset(new BilinearForm(&fes));
    sys.k_coef.reset(new HeterogeneousK(1.0, contrast, center, radius));
    sys.form->AddDomainIntegrator(new DiffusionIntegrator(*sys.k_coef));

    sys.lf.reset(new LinearForm(&fes));
    sys.src_coef.reset(new ConstantCoefficient(1.0));
    sys.lf->AddDomainIntegrator(new DomainLFIntegrator(*sys.src_coef));

    sys.gf.reset(new GridFunction(&fes));
    AssembleAndFormSystem(*sys.form, *sys.lf, fes, mesh,
                          sys.A, sys.X, sys.B, *sys.gf);
    sys.assemble_ms = since(t0);

    r.k_ratio = 1.0; r.peclet = 0.0; r.material_contrast = contrast;
    return sys;
}

// ── 4. 对流-扩散 ────────────────────────────────────────────
AssembledSystem BuildConvDiffHeat(Mesh& mesh, FiniteElementSpace& fes,
                                    double peclet, HeatResult& r)
{
    AssembledSystem sys;
    auto t0 = Clock::now();

    double h = mesh.GetElementSize(0);
    if (h < 1e-12) h = 1e-3;          // v2: 保护退化单元
    double v_mag = 1.0;
    double k_val = v_mag * h / (2.0 * std::max(peclet, 0.01));

    sys.form.reset(new BilinearForm(&fes));
    sys.k_coef.reset(new ConstantCoefficient(k_val));
    sys.form->AddDomainIntegrator(new DiffusionIntegrator(*sys.k_coef));

    int dim = mesh.Dimension();
    Vector vvec(dim); vvec = 0.0; vvec(0) = v_mag;
    if (dim > 1) vvec(1) = 0.5 * v_mag;
    sys.v_coef.reset(new VectorConstantCoefficient(vvec));
    sys.form->AddDomainIntegrator(new ConvectionIntegrator(*sys.v_coef, 1.0));

    sys.lf.reset(new LinearForm(&fes));
    sys.src_coef.reset(new ConstantCoefficient(1.0));
    sys.lf->AddDomainIntegrator(new DomainLFIntegrator(*sys.src_coef));

    sys.gf.reset(new GridFunction(&fes));
    AssembleAndFormSystem(*sys.form, *sys.lf, fes, mesh,
                          sys.A, sys.X, sys.B, *sys.gf);
    sys.assemble_ms = since(t0);

    r.k_ratio = 1.0; r.peclet = peclet; r.material_contrast = 1.0;
    return sys;
}

// ── 5. 罗宾/辐射边界 ────────────────────────────────────────
AssembledSystem BuildRobinHeat(Mesh& mesh, FiniteElementSpace& fes,
                                double h_coef, HeatResult& r)
{
    AssembledSystem sys;
    auto t0 = Clock::now();

    sys.form.reset(new BilinearForm(&fes));
    sys.k_coef.reset(new ConstantCoefficient(1.0));
    sys.form->AddDomainIntegrator(new DiffusionIntegrator(*sys.k_coef));

    sys.robin_coef.reset(new ConstantCoefficient(h_coef));
    sys.form->AddBoundaryIntegrator(new MassIntegrator(*sys.robin_coef));

    sys.lf.reset(new LinearForm(&fes));
    sys.src_coef.reset(new ConstantCoefficient(1.0));
    sys.lf->AddDomainIntegrator(new DomainLFIntegrator(*sys.src_coef));

    sys.gf.reset(new GridFunction(&fes));
    // Robin 边界不需要本质边界条件（全自然边界）
    AssembleAndFormSystem(*sys.form, *sys.lf, fes, mesh,
                          sys.A, sys.X, sys.B, *sys.gf,
                          /*apply_dirichlet_all=*/false);
    sys.assemble_ms = since(t0);

    r.k_ratio = 1.0; r.peclet = 0.0; r.material_contrast = h_coef;
    return sys;
}

// ── 6. 瞬态 隐式 BE 一步 ─────────────────────────────────────
AssembledSystem BuildTransientStep(Mesh& mesh, FiniteElementSpace& fes,
                                    double dt, HeatResult& r)
{
    AssembledSystem sys;
    auto t0 = Clock::now();

    sys.form.reset(new BilinearForm(&fes));
    sys.k_coef.reset(new ConstantCoefficient(1.0));
    sys.form->AddDomainIntegrator(new DiffusionIntegrator(*sys.k_coef));

    sys.inv_dt_coef.reset(new ConstantCoefficient(1.0 / dt));
    sys.form->AddDomainIntegrator(new MassIntegrator(*sys.inv_dt_coef));

    sys.lf.reset(new LinearForm(&fes));
    sys.src_coef.reset(new ConstantCoefficient(1.0));
    sys.lf->AddDomainIntegrator(new DomainLFIntegrator(*sys.src_coef));

    sys.gf.reset(new GridFunction(&fes));
    AssembleAndFormSystem(*sys.form, *sys.lf, fes, mesh,
                          sys.A, sys.X, sys.B, *sys.gf);
    sys.assemble_ms = since(t0);

    r.k_ratio = 1.0 / dt; r.peclet = 0.0; r.material_contrast = dt;
    return sys;
}

// ── 7. 非线性（用平滑解析温度场替代 Randomize）────────────
AssembledSystem BuildNonlinearStep(Mesh& mesh, FiniteElementSpace& fes,
                                    double alpha, HeatResult& r)
{
    AssembledSystem sys;
    auto t0 = Clock::now();

    sys.form.reset(new BilinearForm(&fes));
    sys.k_coef.reset(new SmoothTempDepK(1.0, alpha));  // v2: 解析场
    sys.form->AddDomainIntegrator(new DiffusionIntegrator(*sys.k_coef));

    sys.lf.reset(new LinearForm(&fes));
    sys.src_coef.reset(new ConstantCoefficient(1.0));
    sys.lf->AddDomainIntegrator(new DomainLFIntegrator(*sys.src_coef));

    sys.gf.reset(new GridFunction(&fes));
    AssembleAndFormSystem(*sys.form, *sys.lf, fes, mesh,
                          sys.A, sys.X, sys.B, *sys.gf);
    sys.assemble_ms = since(t0);

    r.k_ratio = 1.0 + alpha; r.peclet = 0.0; r.material_contrast = alpha;
    return sys;
}

// =====================================================================
//  跑一个物理问题 × 求解器列表
// =====================================================================
void RunProblemSuite(Mesh& mesh, FiniteElementSpace& fes,
                      const std::string& problem_name,
                      const std::string& mesh_type,
                      AssembledSystem&& sys, HeatResult base_info,
                      const Config& cfg,
                      std::vector<HeatResult>& all,
                      bool allow_direct)
{
    if (!sys.A.Ptr()) return;
    SparseMatrix* A_sp = dynamic_cast<SparseMatrix*>(sys.A.Ptr());
    if (!A_sp) return;                    // v2: 防御
    SparseMatrix& A = *A_sp;

    MatStats ms;
    try {
        ms = ComputeMatStats(A);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "    [warn] matrix stats failed: %s — 跳过此问题\n",
                     e.what());
        return;
    }

    base_info.n_dof          = fes.GetTrueVSize();
    base_info.n_elements     = mesh.GetNE();
    base_info.nnz            = ms.nnz;
    base_info.nnz_per_row    = ms.nnz_per_row;
    base_info.asymmetry_rel  = ms.asymmetry_rel;
    base_info.asymmetry_abs  = ms.asymmetry_abs;
    base_info.matrix_norm_inf= ms.matrix_norm_inf;
    base_info.is_spd         = ms.is_spd;
    base_info.diag_dominance = ms.diag_dominance;
    base_info.cond_estimate  = ms.cond_estimate;
    base_info.assemble_ms    = sys.assemble_ms;

    auto solvers = GetSolverList(allow_direct &&
                                 base_info.n_dof <= cfg.direct_max_dof);

    int group_start = (int)all.size();
    double best_total = 1e30;
    int    best_idx   = -1;

    for (const auto& spec : solvers) {
        HeatResult r = base_info;
        r.problem      = problem_name;
        r.mesh_type    = mesh_type;
        r.solver_name  = spec.name;
        r.precond_name = spec.precond;

        Vector x(sys.X.Size()); x = 0.0;
        SolveOutput res{false, 0, 0.0, 0.0, 0.0, false};

        if (spec.is_direct) {
#ifdef MFEM_USE_SUITESPARSE
            res = RunDirect(A, sys.B, x);
#else
            continue;
#endif
        } else {
            res = RunIterative(spec.name, spec.precond, A, sys.B, x, cfg,
                                ms.has_zero_diag);
        }

        if (res.skipped) continue;        // v2: 跳过则不记录

        r.converged      = res.converged;
        r.iterations     = res.iters;
        r.final_residual = res.final_res;
        r.solve_ms       = res.solve_ms;
        r.setup_ms       = res.setup_ms;
        r.total_ms       = r.setup_ms + r.solve_ms;

        double b_norm = sys.B.Norml2();
        r.rel_residual = (b_norm > 1e-30) ? r.final_residual / b_norm
                                          : r.final_residual;

        if (r.total_ms > cfg.solver_timeout_s * 1000) r.converged = false;

        if (r.converged && r.total_ms < best_total) {
            best_total = r.total_ms;
            best_idx   = (int)all.size();
        }
        all.push_back(r);
    }

    if (best_idx >= 0) all[best_idx].is_best = true;

    int n_solvers = (int)all.size() - group_start;
    const char* mark = (best_idx >= 0) ? "\033[32m✓\033[0m"
                                        : "\033[31m✗\033[0m";
std::printf("  %s  %-22s %-5s ref=%d  DOF=%7d  nnz=%-8ld  "
                "asym=%.4f  cond=1e%.1f  最优:%s (%.1fms)\n",
                mark, problem_name.c_str(), mesh_type.c_str(),
                base_info.ref_level, base_info.n_dof, ms.nnz,
                ms.asymmetry_rel, ms.cond_estimate,
                best_idx >= 0 ? all[best_idx].solver_name.c_str() : "—",
                best_idx >= 0 ? best_total : 0.0);

    if (cfg.verbose && n_solvers > 0) {
        std::vector<int> idx(n_solvers);
        for (int i = 0; i < n_solvers; ++i) idx[i] = group_start + i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b){
            if (all[a].converged != all[b].converged) return all[a].converged;
            return all[a].total_ms < all[b].total_ms;
        });
        std::printf("      ");
        for (int k = 0; k < std::min(n_solvers, 3); ++k) {
            const auto& rr = all[idx[k]];
            std::printf("[%d]%s%s %.1fms/%dit  ",
                        k+1, rr.solver_name.c_str(),
                        rr.converged ? "✓" : "✗",
                        rr.total_ms, rr.iterations);
        }
        std::printf("\n");
    }
}

// =====================================================================
//  网格生成（v2：三角/四面体起始网格用 2x2 / 2x2x2 避免退化）
// =====================================================================
std::shared_ptr<Mesh> MakeMesh(int dim, Element::Type type, int ref)
{
    std::shared_ptr<Mesh> mesh;

    // v2: 三角/四面体从更细的起始网格开始，避免 UniformRefinement 时的数值退化
    int n0 = 1;
    if (type == Element::TRIANGLE || type == Element::TETRAHEDRON) n0 = 2;

    if (dim == 2) {
        mesh = std::make_shared<Mesh>(
            Mesh::MakeCartesian2D(n0, n0, type, /*generate_edges=*/true,
                                   1.0, 1.0));
    } else {
        mesh = std::make_shared<Mesh>(
            Mesh::MakeCartesian3D(n0, n0, n0, type, 1.0, 1.0, 1.0));
    }

    // v2: 首次 finalize，非共形网格需要
    mesh->EnsureNodes();

    for (int i = 0; i < ref; ++i) mesh->UniformRefinement();
    return mesh;
}

std::string MeshTypeName(Element::Type t) {
    switch (t) {
        case Element::QUADRILATERAL: return "QUAD";
        case Element::TRIANGLE:      return "TRI";
        case Element::HEXAHEDRON:    return "HEX";
        case Element::TETRAHEDRON:   return "TET";
        default:                      return "OTHER";
    }
}

// =====================================================================
//  CSV 输出
// =====================================================================
void WriteCSVHeader(std::ofstream& f) {
    f << "problem,mesh_type,dim,ref_level,poly_order,n_elements,"
         "n_dof,nnz,nnz_per_row,"
         "asymmetry_rel,asymmetry_abs,matrix_norm_inf,"
         "is_spd,diag_dominance,log10_cond_estimate,"
         "k_ratio,peclet,material_contrast,"
         "solver,precond,"
         "assemble_ms,setup_ms,solve_ms,total_ms,"
         "converged,iterations,final_residual,rel_residual,is_best\n";
}
 
void WriteCSVRow(std::ofstream& f, const HeatResult& r) {
    f << r.problem         << ","
      << r.mesh_type       << ","
      << r.dim             << ","
      << r.ref_level       << ","
      << r.poly_order      << ","
      << r.n_elements      << ","
      << r.n_dof           << ","
      << r.nnz             << ","
      << std::fixed << std::setprecision(4)
      << r.nnz_per_row     << ","
      << std::scientific << std::setprecision(4)
      << r.asymmetry_rel   << ","
      << r.asymmetry_abs   << ","
      << r.matrix_norm_inf << ","
      << std::fixed << std::setprecision(4)
      << (r.is_spd ? 1 : 0) << ","
      << r.diag_dominance  << ","
      << r.cond_estimate   << ","
      << r.k_ratio         << ","
      << r.peclet          << ","
      << r.material_contrast << ","
      << r.solver_name     << ","
      << r.precond_name    << ","
      << std::setprecision(3)
      << r.assemble_ms     << ","
      << r.setup_ms        << ","
      << r.solve_ms        << ","
      << r.total_ms        << ","
      << (r.converged ? 1 : 0) << ","
      << r.iterations      << ","
      << std::scientific << std::setprecision(4)
      << r.final_residual  << ","
      << r.rel_residual    << ","
      << (r.is_best ? 1 : 0)
      << "\n";
}

// =====================================================================
//  主函数
// =====================================================================
int main(int argc, char** argv)
{
    Config cfg = ParseArgs(argc, argv);

    std::printf("\n\033[1m╔══════════════════════════════════════════════════════════════╗\n"
                  "║       MFEM 热传导问题 — 最优求解器数据收集 (v2)              ║\n"
                  "╚══════════════════════════════════════════════════════════════╝\033[0m\n");
    std::printf("  FEM 阶数  : P%d\n", cfg.order);
    std::printf("  收敛容差  : %.0e (rel) / %.0e (abs)\n", cfg.rtol, cfg.atol);
    std::printf("  最大迭代  : %d\n", cfg.max_iter);
    std::printf("  超时      : %.1f s\n", cfg.solver_timeout_s);
    std::printf("  直接法上限: %d DOF\n", cfg.direct_max_dof);
    std::printf("  输出      : %s  (最优子集: %s)\n",
                cfg.output_all.c_str(), cfg.output_best.c_str());

    std::vector<HeatResult> all;
    all.reserve(4096);

#ifdef MFEM_USE_SUITESPARSE
    bool has_direct = !cfg.skip_direct;
    std::printf("  UMFPACK   : 已启用\n");
#else
    bool has_direct = false;
    std::printf("  UMFPACK   : 未编译\n");
#endif

    // 构造所有 (dim, 单元类型, 精炼范围) 组合
    struct MeshCase { int dim; Element::Type type; int min_ref, max_ref; };
    std::vector<MeshCase> mesh_cases;
    if (cfg.run_2d) {
        if (cfg.run_quad)
            mesh_cases.push_back({2, Element::QUADRILATERAL,
                                   cfg.min_ref_2d, cfg.max_ref_2d});
        if (cfg.run_tri)
            mesh_cases.push_back({2, Element::TRIANGLE,
                                   cfg.min_ref_2d, cfg.max_ref_2d});
    }
    if (cfg.run_3d) {
        if (cfg.run_hex)
            mesh_cases.push_back({3, Element::HEXAHEDRON,
                                   cfg.min_ref_3d, cfg.max_ref_3d});
        if (cfg.run_tet)
            mesh_cases.push_back({3, Element::TETRAHEDRON,
                                   cfg.min_ref_3d, cfg.max_ref_3d});
    }

    // 统一的"跑一个物理问题"包装（每次新建 FES）
    auto run_case = [&](const std::string& pname, Mesh& mesh, int ref,
                         const MeshCase& mc,
                         std::function<AssembledSystem(
                             FiniteElementSpace&, HeatResult&)> builder) {
        try {
            H1_FECollection fec(cfg.order, mc.dim);
            FiniteElementSpace fes(&mesh, &fec);
            HeatResult base;
            base.dim        = mc.dim;
            base.ref_level  = ref;
            base.poly_order = cfg.order;
            auto sys = builder(fes, base);
            if (!sys.valid) return;
            RunProblemSuite(mesh, fes, pname, MeshTypeName(mc.type),
                             std::move(sys), base, cfg, all, has_direct);
        } catch (const std::exception& e) {
            std::fprintf(stderr,
                "  [skip] %s (%s ref=%d) 组装失败: %s\n",
                pname.c_str(), MeshTypeName(mc.type).c_str(), ref, e.what());
        }
    };

    // ── 遍历：网格 × 精炼级 × 物理问题 × 求解器 ─────────────
    for (const auto& mc : mesh_cases) {
        for (int ref = mc.min_ref; ref <= mc.max_ref; ++ref) {
            std::shared_ptr<Mesh> mesh;
            try {
                mesh = MakeMesh(mc.dim, mc.type, ref);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "  [skip] mesh %dD %s ref=%d: %s\n",
                             mc.dim, MeshTypeName(mc.type).c_str(), ref,
                             e.what());
                continue;
            }

            std::printf("\n\033[36m── %dD %s  ref=%d  (elements=%d) ──\033[0m\n",
                        mc.dim, MeshTypeName(mc.type).c_str(),
                        ref, mesh->GetNE());

            // 1. 稳态均匀（不同 k 值）
            if (cfg.run_steady) {
                std::vector<double> x = {1.0, 5.0 ,10.0, 50.0 , 100.0};
                for (double k_val : x) {
                    std::string pname = "SteadyHeat_k" + std::to_string((int)k_val);
                    run_case(pname, *mesh, ref, mc,
                        [&](FiniteElementSpace& fes, HeatResult& r){
                            return BuildSteadyHeat(*mesh, fes, k_val, r);
                        });
                }
            }

            // 2. 各向异性
            if (cfg.run_aniso) {
                std::vector<std::tuple<double,double,double>> cases = {
                    {10.0, 1.0, 1.0},
                    {100.0, 1.0, 1.0},
                    {1.0, 100.0, 1.0},
                    {0.1, 17.0,  123.0},
                    {5.0, 1000.0, 500.0},
                };
                for (auto [kx, ky, kz] : cases) {
                    std::string pname = "AnisoHeat_"+std::to_string((int)kx)
                                         + "_" + std::to_string((int)ky);
                    run_case(pname, *mesh, ref, mc,
                        [&](FiniteElementSpace& fes, HeatResult& r){
                            return BuildAnisoHeat(*mesh, fes, kx, ky, kz, r);
                        });
                }
            }

            // 3. 多材料
            if (cfg.run_multi_mat) {
                for (double c : {10.0, 50.0, 100.0, 500.0, 1000.0, 1e6}) {
                    std::ostringstream oss;
                    oss << "MultiMat_c1e" << (int)std::log10(c);
                    run_case(oss.str(), *mesh, ref, mc,
                        [&](FiniteElementSpace& fes, HeatResult& r){
                            return BuildMultiMatHeat(*mesh, fes, c, r);
                        });
                }
            }

            // // 4. 对流-扩散
            // if (cfg.run_convdiff) {
            //     for (double pe : {1.0, 10.0, 100.0}) {
            //         std::string pname = "ConvDiff_Pe" + std::to_string((int)pe);
            //         run_case(pname, *mesh, ref, mc,
            //             [&](FiniteElementSpace& fes, HeatResult& r){
            //                 return BuildConvDiffHeat(*mesh, fes, pe, r);
            //             });
            //     }
            // }

            // 5. 罗宾/辐射边界
            if (cfg.run_robin) {
                for (double hc : {0.1, 1.0, 10.0, 50.0, 100.0, 500.0 ,1000.0}) {
                    std::ostringstream oss;
                    oss << "RobinHeat_h" << (int)hc;
                    run_case(oss.str(), *mesh, ref, mc,
                        [&](FiniteElementSpace& fes, HeatResult& r){
                            return BuildRobinHeat(*mesh, fes, hc, r);
                        });
                }
            }

            // 6. 瞬态（隐式 BE）
            if (cfg.run_transient) {
                for (double dt : {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}) {
                    std::ostringstream oss;
                    oss << "Transient_dt" << std::scientific
                        << std::setprecision(0) << dt;
                    run_case(oss.str(), *mesh, ref, mc,
                        [&](FiniteElementSpace& fes, HeatResult& r){
                            return BuildTransientStep(*mesh, fes, dt, r);
                        });
                }
            }

            // 7. 非线性 Newton 线性化步
            if (cfg.run_nonlinear) {
                for (double alpha : {0.1, 1.0, 5.0, 10.0, 50.0, 100.0}) {
                    std::ostringstream oss;
                    oss << "Nonlinear_a" << std::fixed
                        << std::setprecision(1) << alpha;
                    run_case(oss.str(), *mesh, ref, mc,
                        [&](FiniteElementSpace& fes, HeatResult& r){
                            return BuildNonlinearStep(*mesh, fes, alpha, r);
                        });
                }
            }
        }
    }

    // ── 写 CSV ────────────────────────────────────────────────
    std::ofstream f_all(cfg.output_all), f_best(cfg.output_best);
    if (!f_all.is_open()) {
        std::fprintf(stderr, "[ERR] 无法打开 %s\n", cfg.output_all.c_str());
        return 1;
    }
    WriteCSVHeader(f_all);
    WriteCSVHeader(f_best);
    for (const auto& r : all) {
        WriteCSVRow(f_all, r);
        if (r.is_best) WriteCSVRow(f_best, r);
    }
    f_all.close(); f_best.close();

    // ── 汇总统计 ──────────────────────────────────────────────
    std::printf("\n\033[1m═══ 汇总 ═══\033[0m\n");
    std::printf("  总记录数  : %zu\n", all.size());

    std::map<std::string, int> wins, converge, total;
    for (const auto& r : all) {
        total[r.solver_name]++;
        if (r.converged) converge[r.solver_name]++;
        if (r.is_best)   wins[r.solver_name]++;
    }

    std::printf("\n  %-18s %8s %10s %10s\n",
                "求解器", "胜场", "收敛率", "尝试");
    std::printf("  %s\n", std::string(55,'-').c_str());

    std::set<std::string> all_solvers;
    for (auto& [k,v] : total) all_solvers.insert(k);
    for (auto& k : all_solvers) {
        int w  = wins.count(k)     ? wins[k]     : 0;
        int cv = converge.count(k) ? converge[k] : 0;
        int tt = total[k];
        double cr = (tt > 0) ? 100.0 * cv / tt : 0.0;
        std::printf("  %-18s %8d %9.1f%% %10d\n", k.c_str(), w, cr, tt);
    }

    std::printf("\n  \033[32m✅ 已写入:\033[0m\n");
    std::printf("    详细记录 : %s\n", cfg.output_all.c_str());
    std::printf("    最优子集 : %s  (可直接作为 ML 训练数据)\n",
                cfg.output_best.c_str());
    return 0;
}