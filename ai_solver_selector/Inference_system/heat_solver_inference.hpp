#pragma once
/**
 * heat_solver_inference.hpp
 * =========================
 * 基于 PyTorch 训练好的 MLP，为 MFEM 热传导问题自动选择最优求解器。
 *
 * 功能：
 *   1. 从 MFEM Mesh + FiniteElementSpace + 组装好的 SparseMatrix
 *      中提取 30+ 维特征向量（与 train_heat_solver.py 完全对齐）
 *   2. 加载 solver_heat_model.json（纯 C++ MLP 推理，零 ML 库依赖）
 *   3. 创建配置好的 MFEM Solver 对象
 *   4. 可选：运行求解并返回结果指标
 *
 * 依赖：
 *   - MFEM（serial）
 *   - nlohmann/json（json.hpp 单头文件）
 *   - mlp_inference.hpp（MLP 前向推理）
 *   - C++17
 *
 * 使用：
 *   HeatSolverSelector sel("models/solver_heat_model.json");
 *
 *   // 选项 A：直接给描述好的特征
 *   HeatProblemDesc desc;
 *   desc.problem_family = "SteadyHeat";
 *   desc.k_ratio = 100.0;      // 各向异性比
 *   desc.peclet  = 0.0;
 *   auto pick = sel.Pick(A, mesh, fespace, desc);
 *
 *   // 选项 B：从矩阵 + 网格自动推断问题族
 *   auto pick = sel.PickAuto(A, mesh, fespace);
 *
 *   // 直接求解
 *   auto result = sel.SolveAuto(A, b, x, mesh, fespace);
 */

#ifndef HEAT_SOLVER_INFERENCE_HPP
#define HEAT_SOLVER_INFERENCE_HPP

#include "mfem.hpp"
#include "mlp_inference.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// ======================================================================
//  特征 / 求解器枚举
// ======================================================================

/// 热传导问题家族（必须与 Python 训练数据中的 problem_family 对齐）
/// 这些字符串必须与 collect_heat_data.cpp 中生成 problem 名时 split("_")[0]
/// 的结果完全一致。
enum class ProblemFamily {
    SteadyHeat,
    AnisoHeat,
    MultiMat,
    ConvDiff,
    RobinHeat,
    Transient,
    Nonlinear,
    ReactDiff,    // v3.1: (K + r·M) u = f
    AnisoMat,     // v3.1: anisotropy + multi-material
    ConvAniso,    // v3.1: convection + anisotropy
    Unknown,
};

inline const char* ProblemFamilyName(ProblemFamily p) {
    switch (p) {
        case ProblemFamily::SteadyHeat: return "SteadyHeat";
        case ProblemFamily::AnisoHeat:  return "AnisoHeat";
        case ProblemFamily::MultiMat:   return "MultiMat";
        case ProblemFamily::ConvDiff:   return "ConvDiff";
        case ProblemFamily::RobinHeat:  return "RobinHeat";
        case ProblemFamily::Transient:  return "Transient";
        case ProblemFamily::Nonlinear:  return "Nonlinear";
        case ProblemFamily::ReactDiff:  return "ReactDiff";
        case ProblemFamily::AnisoMat:   return "AnisoMat";
        case ProblemFamily::ConvAniso:  return "ConvAniso";
        default:                         return "Unknown";
    }
}

inline ProblemFamily ParseProblemFamily(const std::string& s) {
    if (s == "SteadyHeat") return ProblemFamily::SteadyHeat;
    if (s == "AnisoHeat")  return ProblemFamily::AnisoHeat;
    if (s == "MultiMat")   return ProblemFamily::MultiMat;
    if (s == "ConvDiff")   return ProblemFamily::ConvDiff;
    if (s == "RobinHeat")  return ProblemFamily::RobinHeat;
    if (s == "Transient")  return ProblemFamily::Transient;
    if (s == "Nonlinear")  return ProblemFamily::Nonlinear;
    if (s == "ReactDiff")  return ProblemFamily::ReactDiff;
    if (s == "AnisoMat")   return ProblemFamily::AnisoMat;
    if (s == "ConvAniso")  return ProblemFamily::ConvAniso;
    return ProblemFamily::Unknown;
}

/// 用户描述的问题信息（神经网络不能从矩阵单独推断的部分）
struct HeatProblemDesc {
    ProblemFamily problem_family = ProblemFamily::SteadyHeat;
    double k_ratio           = 1.0;   ///< 各向异性比 max(k)/min(k)
    double peclet            = 0.0;   ///< Péclet 数（仅 ConvDiff）
    double material_contrast = 1.0;   ///< 多材料导热比
};

// ======================================================================
//  矩阵特征（与 heat_benchmark.cpp 的 MatStats 完全对齐）
// ======================================================================
struct HeatMatFeatures {
    long   nnz             = 0;
    double nnz_per_row     = 0.0;
    double asymmetry_rel   = 0.0;
    double asymmetry_abs   = 0.0;
    double matrix_norm_inf = 0.0;
    bool   is_spd          = true;
    double diag_dominance  = 1.0;
    double log10_cond_estimate = 0.0;

    void Print() const {
        std::printf("  nnz=%-8ld nnz/row=%.2f  asym_rel=%.4e  asym_abs=%.2e\n"
                    "  ||A||_inf=%.2e  is_spd=%d  diag_dom=%.2f  log10(cond)=%.2f\n",
                    nnz, nnz_per_row, asymmetry_rel, asymmetry_abs,
                    matrix_norm_inf, is_spd ? 1 : 0,
                    diag_dominance, log10_cond_estimate);
    }
};

// ======================================================================
//  推理结果
// ======================================================================
struct HeatSolverPick {
    std::string best_solver;                             ///< 推荐求解器名（如 "PCG_Cheby"）
    float       confidence = 0.0f;                       ///< 置信度 [0, 1]
    std::vector<std::pair<std::string, float>> ranking;  ///< 按置信度降序的完整排名
    HeatMatFeatures mat_features;                        ///< 调试用

    void Print() const {
        std::printf("  推荐求解器: \033[1;32m%s\033[0m  (置信度 %.1f%%)\n",
                    best_solver.c_str(), confidence * 100.0f);
        int limit = std::min<int>(4, (int)ranking.size());
        std::printf("  排名: ");
        for (int i = 0; i < limit; ++i) {
            std::printf("[%d]%s %.0f%%  ", i + 1,
                        ranking[i].first.c_str(),
                        ranking[i].second * 100.0f);
        }
        std::printf("\n");
    }
};

// ======================================================================
//  求解结果（Solve 接口返回）
// ======================================================================
struct HeatSolveResult {
    std::string solver_used;
    bool   converged      = false;
    int    iterations     = 0;
    double final_residual = 0.0;
    double setup_ms       = 0.0;
    double solve_ms       = 0.0;
    double total_ms       = 0.0;
};

// ======================================================================
//  HeatSolverSelector 主类
// ======================================================================
class HeatSolverSelector {
public:
    explicit HeatSolverSelector(const std::string& model_path) {
        mlp_.LoadFromFile(model_path);

        // 从模型 JSON 中取回 feature_names 顺序 (mlp_ 已持有)
        feature_names_ = mlp_.feature_names;
        class_names_   = mlp_.class_names;

        // 建立 feature_name → index 映射
        for (int i = 0; i < (int)feature_names_.size(); ++i)
            feat_index_[feature_names_[i]] = i;

        std::printf("[HeatSolverSelector] 已加载: %s\n", model_path.c_str());
        std::printf("  特征维度: %d  类别数: %d\n",
                    mlp_.n_features, mlp_.n_classes);
        std::printf("  归一化策略: %s", mlp_.ScalerKind().c_str());
        if (mlp_.ClipEnabled()) {
            std::printf("  (clip=[%.2f, %.2f])", mlp_.ClipLo(), mlp_.ClipHi());
        }
        std::printf("\n  支持求解器: ");
        for (const auto& c : class_names_) std::printf("%s ", c.c_str());
        std::printf("\n");
    }

    void SetVerbose(bool v) { verbose_ = v; }

    // ── 特征提取 ─────────────────────────────────────────────

    /// 从稀疏矩阵计算矩阵特征（与 heat_benchmark.cpp::ComputeMatStats 对齐）
    static HeatMatFeatures ExtractMatFeatures(const mfem::SparseMatrix& A) {
        HeatMatFeatures f;
        int N = A.Height();
        f.nnz         = A.NumNonZeroElems();
        f.nnz_per_row = (double)f.nnz / std::max(N, 1);

        const int*    I = A.GetI();
        const int*    J = A.GetJ();
        const double* V = A.GetData();

        auto CSR_Get = [&](int row, int col) -> double {
            for (int k = I[row]; k < I[row + 1]; ++k)
                if (J[k] == col) return V[k];
            return 0.0;
        };

        // 非对称度（全扫描）
        double asym_acc = 0.0, mag_acc = 0.0, asym_max = 0.0, row_sum_max = 0.0;
        for (int row = 0; row < N; ++row) {
            double row_abs_sum = 0.0;
            for (int k = I[row]; k < I[row + 1]; ++k) {
                double v = V[k];
                row_abs_sum += std::abs(v);
                int col = J[k];
                if (col == row) continue;
                double aji   = CSR_Get(col, row);
                double diff  = std::abs(v - aji);
                double denom = std::abs(v) + std::abs(aji);
                asym_acc += diff;
                mag_acc  += denom;
                asym_max  = std::max(asym_max, diff);
            }
            row_sum_max = std::max(row_sum_max, row_abs_sum);
        }
        f.asymmetry_rel   = (mag_acc > 1e-30) ? asym_acc / mag_acc : 0.0;
        f.asymmetry_abs   = asym_max;
        f.matrix_norm_inf = row_sum_max;

        // SPD + 对角优势
        int spd_cnt = 0;
        double dd_acc = 0.0, diag_min = 1e30;
        for (int row = 0; row < N; ++row) {
            double diag = 0.0, off = 0.0;
            for (int k = I[row]; k < I[row + 1]; ++k) {
                if (J[k] == row) diag = V[k];
                else             off += std::abs(V[k]);
            }
            if (diag > off) ++spd_cnt;
            double dd = (off > 1e-30) ? diag / off : 10.0;
            dd_acc += std::min(dd, 10.0);
            double d = std::abs(diag);
            if (d > 1e-30) diag_min = std::min(diag_min, d);
        }
        f.is_spd         = (spd_cnt == N) && (f.asymmetry_rel < 0.01);
        f.diag_dominance = dd_acc / std::max(N, 1);

        // 条件数粗估（功率迭代）
        mfem::Vector v(N), w(N);
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
        f.log10_cond_estimate = std::log10(std::max(1.0, cond));
        return f;
    }

    // ── 推理 ─────────────────────────────────────────────────

    /**
     * @brief 给定问题描述 + MFEM 对象，返回推荐求解器
     */
    HeatSolverPick Pick(const mfem::SparseMatrix& A,
                         const mfem::Mesh& mesh,
                         const mfem::FiniteElementSpace& fespace,
                         const HeatProblemDesc& desc) const
    {
        HeatSolverPick pick;
        pick.mat_features = ExtractMatFeatures(A);

        auto feat_vec = BuildFeatureVector(A, mesh, fespace,
                                            desc, pick.mat_features);
        auto proba = mlp_.Predict(feat_vec);

        int best = (int)(std::max_element(proba.begin(), proba.end())
                          - proba.begin());
        pick.best_solver = class_names_[best];
        pick.confidence  = proba[best];

        // 排名
        std::vector<std::pair<std::string, float>> ranked;
        ranked.reserve(proba.size());
        for (size_t i = 0; i < proba.size(); ++i)
            ranked.emplace_back(class_names_[i], proba[i]);
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& a, const auto& b){ return a.second > b.second; });
        pick.ranking = std::move(ranked);

        if (verbose_) {
            pick.mat_features.Print();
            pick.Print();
        }
        return pick;
    }

    /**
     * @brief 自动推断问题族（从矩阵对称性 + 对角优势启发式）并推理
     */
    HeatSolverPick PickAuto(const mfem::SparseMatrix& A,
                             const mfem::Mesh& mesh,
                             const mfem::FiniteElementSpace& fespace) const
    {
        auto mat_f = ExtractMatFeatures(A);
        HeatProblemDesc desc;

        // 启发式判断
        if (mat_f.asymmetry_rel > 0.05) {
            desc.problem_family = ProblemFamily::ConvDiff;
            desc.peclet = 10.0;   // 默认中等 Pe
        } else if (mat_f.asymmetry_rel < 0.01 && mat_f.is_spd) {
            desc.problem_family = ProblemFamily::SteadyHeat;
        } else {
            desc.problem_family = ProblemFamily::SteadyHeat;
        }

        return Pick(A, mesh, fespace, desc);
    }

    // ── 求解器工厂 ───────────────────────────────────────────

    /**
     * @brief 根据求解器名称创建配置好的 MFEM Solver，并求解 A x = b
     *
     * 支持的求解器（必须与 collect_heat_data.cpp::GetSolverList 对齐）：
     *   - CG / PCG_Jacobi / PCG_l1Jac / PCG_GS / PCG_Cheby
     *   - MINRES / MINRES_Jac
     *   - GMRES / GMRES_Jac / GMRES_GS
     *   - FGMRES_Jac / FGMRES_GS
     *   - BiCGSTAB / BiCGSTAB_Jac / BiCGSTAB_GS
     *   - SLI_Jac (可选)
     *   - DIRECT_UMF (若编译开启 SuiteSparse)
     */
    HeatSolveResult Solve(const std::string& solver_name,
                           mfem::SparseMatrix& A,
                           const mfem::Vector& b, mfem::Vector& x,
                           double rtol = 1e-8, int max_iter = 2000) const
    {
        using Clock = std::chrono::high_resolution_clock;
        auto ms_since = [](Clock::time_point t){
            return std::chrono::duration<double, std::milli>(
                Clock::now() - t).count();
        };

        HeatSolveResult r;
        r.solver_used = solver_name;

        // ── 解析名字尾部的前提条件子后缀 ───────────────────────
        // "PCG_Jacobi" → "Jacobi" ；"GMRES" → "" ； "BiCGSTAB_GS" → "GS"
        std::string prec_suffix;
        {
            auto p = solver_name.find('_');
            if (p != std::string::npos) prec_suffix = solver_name.substr(p + 1);
        }

        std::unique_ptr<mfem::Solver> prec;
        auto t_setup = Clock::now();

        // 预条件器
        if      (prec_suffix == "GS")       prec.reset(new mfem::GSSmoother(A));
        else if (prec_suffix == "Jacobi" ||
                 prec_suffix == "Jac")      prec.reset(new mfem::DSmoother(A, 0));
        else if (prec_suffix == "l1Jac")    prec.reset(new mfem::DSmoother(A, 1));
        else if (prec_suffix == "Cheby")    prec.reset(new mfem::DSmoother(A, 2, 10));
        r.setup_ms = ms_since(t_setup);

        // 迭代求解器（FGMRES 必须先于 GMRES 匹配）
        std::unique_ptr<mfem::IterativeSolver> solver;
        if (solver_name == "CG" || solver_name.substr(0, 3) == "PCG") {
            solver.reset(new mfem::CGSolver());
        } else if (solver_name.substr(0, 6) == "MINRES") {
            solver.reset(new mfem::MINRESSolver());
        } else if (solver_name.substr(0, 6) == "FGMRES") {
            auto* fg = new mfem::FGMRESSolver();
            fg->SetKDim(30);
            solver.reset(fg);
        } else if (solver_name.substr(0, 5) == "GMRES") {
            auto* gm = new mfem::GMRESSolver();
            gm->SetKDim(30);
            solver.reset(gm);
        } else if (solver_name.substr(0, 8) == "BiCGSTAB") {
            solver.reset(new mfem::BiCGSTABSolver());
        } else if (solver_name.substr(0, 3) == "SLI") {
            solver.reset(new mfem::SLISolver());
        }
#ifdef MFEM_USE_SUITESPARSE
        else if (solver_name == "DIRECT_UMF") {
            mfem::UMFPackSolver direct;
            direct.SetOperator(A);
            r.setup_ms = ms_since(t_setup);

            auto t_solve = Clock::now();
            x = 0.0;
            direct.Mult(b, x);
            r.solve_ms  = ms_since(t_solve);
            r.total_ms  = r.setup_ms + r.solve_ms;
            r.converged = true;
            r.iterations = 1;
            r.final_residual = 0.0;
            return r;
        }
#endif
        else {
            std::fprintf(stderr, "[HeatSolverSelector] 未知求解器: %s\n",
                         solver_name.c_str());
            return r;
        }

        solver->SetOperator(A);
        if (prec) solver->SetPreconditioner(*prec);
        solver->SetRelTol(rtol);
        solver->SetMaxIter(max_iter);
        solver->SetPrintLevel(verbose_ ? 1 : -1);

        x = 0.0;
        auto t_solve = Clock::now();
        solver->Mult(b, x);
        r.solve_ms  = ms_since(t_solve);
        r.total_ms  = r.setup_ms + r.solve_ms;
        r.converged      = solver->GetConverged();
        r.iterations     = solver->GetNumIterations();
        r.final_residual = solver->GetFinalNorm();

        if (!std::isfinite(r.final_residual)) r.converged = false;
        return r;
    }

    /**
     * @brief 一步到位：推理 → 求解
     */
    HeatSolveResult SolveAuto(mfem::SparseMatrix& A,
                                const mfem::Vector& b, mfem::Vector& x,
                                const mfem::Mesh& mesh,
                                const mfem::FiniteElementSpace& fespace,
                                const HeatProblemDesc& desc,
                                double rtol = 1e-8, int max_iter = 2000) const
    {
        auto pick = Pick(A, mesh, fespace, desc);
        if (verbose_) {
            std::printf("[HeatSolverSelector] 自动选择: %s\n",
                        pick.best_solver.c_str());
        }
        return Solve(pick.best_solver, A, b, x, rtol, max_iter);
    }

    const std::vector<std::string>& ClassNames() const { return class_names_; }

private:
    MlpSolverModel           mlp_;
    std::vector<std::string> feature_names_;
    std::vector<std::string> class_names_;
    std::unordered_map<std::string, int> feat_index_;
    bool                     verbose_ = false;

    // ── 将 MFEM 对象 + 问题描述 → 特征向量（顺序与训练端完全一致）──
    std::vector<double> BuildFeatureVector(
        const mfem::SparseMatrix& A,
        const mfem::Mesh& mesh,
        const mfem::FiniteElementSpace& fespace,
        const HeatProblemDesc& desc,
        const HeatMatFeatures& mf) const
    {
        int n_feat = (int)feature_names_.size();
        std::vector<double> v(n_feat, 0.0);

        // Helper：如果 feature_names_ 中有该名，就设置值
        auto set = [&](const std::string& name, double val){
            auto it = feat_index_.find(name);
            if (it != feat_index_.end()) v[it->second] = val;
        };

        // 原始数值特征（与 NUMERIC_FEATURES 对应）
        int dim        = mesh.Dimension();
        int n_elements = mesh.GetNE();
        int n_dof      = fespace.GetTrueVSize();
        int poly_order = fespace.GetMaxElementOrder();

        set("dim",                  (double)dim);
        set("ref_level",             0.0);   // 推理时未知，填 0
        set("poly_order",           (double)poly_order);
        set("n_elements",           (double)n_elements);
        set("n_dof",                (double)n_dof);
        set("nnz",                  (double)mf.nnz);
        set("nnz_per_row",          mf.nnz_per_row);
        set("asymmetry_rel",        mf.asymmetry_rel);
        set("asymmetry_abs",        mf.asymmetry_abs);
        set("matrix_norm_inf",      mf.matrix_norm_inf);
        set("is_spd",               mf.is_spd ? 1.0 : 0.0);
        set("diag_dominance",       mf.diag_dominance);
        set("log10_cond_estimate",  mf.log10_cond_estimate);
        set("k_ratio",              desc.k_ratio);
        set("peclet",               desc.peclet);
        set("material_contrast",    desc.material_contrast);

        // 派生特征（与 engineer_features 对应）
        set("log_n_dof",         std::log10(std::max(1, n_dof)));
        set("log_n_elements",    std::log10(std::max(1, n_elements)));
        set("log_nnz",           std::log10(std::max(1L, mf.nnz)));
        set("log_k_ratio",       std::log10(std::max(1e-6, desc.k_ratio)));
        set("log_peclet",        std::log10(std::max(1e-6, desc.peclet)));
        set("log_mat_contrast",  std::log10(std::max(1e-6, desc.material_contrast)));
        set("log_norm_inf",      std::log10(std::max(1e-6, mf.matrix_norm_inf)));

        set("is_3d",             (dim == 3)                   ? 1.0 : 0.0);
        set("is_high_order",     (poly_order > 1)             ? 1.0 : 0.0);
        set("is_anisotropic",    (desc.k_ratio > 5)           ? 1.0 : 0.0);
        set("is_convection",     (desc.peclet > 1)            ? 1.0 : 0.0);
        set("is_heterogeneous",  (desc.material_contrast > 10)? 1.0 : 0.0);
        set("is_large",          (std::log10(std::max(1, n_dof)) > 4.5) ? 1.0 : 0.0);

        set("complexity",        std::log10(std::max(1, n_dof)) + mf.log10_cond_estimate);
        set("sym_score",         1.0 - mf.asymmetry_rel);

        // One-hot: mesh_type
        std::string mesh_tag;
        auto type0 = (mesh.GetNE() > 0) ? mesh.GetElementType(0)
                                         : mfem::Element::QUADRILATERAL;
        switch (type0) {
            case mfem::Element::QUADRILATERAL: mesh_tag = "mesh_QUAD"; break;
            case mfem::Element::TRIANGLE:      mesh_tag = "mesh_TRI";  break;
            case mfem::Element::HEXAHEDRON:    mesh_tag = "mesh_HEX";  break;
            case mfem::Element::TETRAHEDRON:   mesh_tag = "mesh_TET";  break;
            default:                            mesh_tag = "";
        }
        if (!mesh_tag.empty()) set(mesh_tag, 1.0);

        // One-hot: problem_family
        std::string phys_tag = std::string("phys_") +
                                ProblemFamilyName(desc.problem_family);
        set(phys_tag, 1.0);

        return v;
    }
};

#endif // HEAT_SOLVER_INFERENCE_HPP