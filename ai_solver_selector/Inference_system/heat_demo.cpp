/**
 * heat_demo.cpp
 * =============
 * 演示：用训练好的 PyTorch MLP 模型，在 MFEM 热传导问题中自动选择最优求解器。
 *
 * 工作流：
 *   1. 构建 MFEM 网格 + 热传导问题（与 heat_benchmark.cpp 中相同的物理模型）
 *   2. 加载 solver_heat_model.json
 *   3. 提取特征 → MLP 推理 → 获得推荐求解器
 *   4. 创建求解器并实际求解，打印性能
 *   5. （可选）对比所有求解器，验证 AI 推荐是否真的最优
 *
 * 编译：
 *   g++ -std=c++17 -O3 \
 *       -I$(MFEM_DIR)/include -I./ \
 *       heat_demo.cpp \
 *       -L$(MFEM_DIR)/lib -lmfem -lm \
 *       -o heat_demo
 *
 * 运行：
 *   ./heat_demo --model ../models/solver_heat_model.json \
 *               --problem steady --ref 4
 *
 *   ./heat_demo --model ../models/solver_heat_model.json \
 *               --problem aniso --k 100 --ref 5 --compare
 *
 *   ./heat_demo --model ../models/solver_heat_model.json \
 *               --problem convdiff --peclet 100 --ref 4 --compare
 */

#include "heat_solver_inference.hpp"

#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace mfem;

// ──────────────────────────────────────────────────────────────
// 命令行配置
// ──────────────────────────────────────────────────────────────
struct DemoConfig {
    std::string model_path = "/home/ych/yfem-p3/ai_solver_selector/torch_training/models/solver_heat_model.json";
    std::string problem    = "steady";   // steady|aniso|multimat|convdiff|robin
    int         dim        = 2;
    std::string mesh_type  = "quad";     // quad|tri|hex|tet
    int         ref        = 3;
    int         order      = 1;

    double k_value          = 1.0;
    double k_ratio          = 1.0;       // 各向异性比
    double peclet           = 1.0;
    double material_contrast = 1.0;
    double robin_h          = 1.0;

    double rtol    = 1e-8;
    int    maxit   = 2000;
    bool   compare = false;
    bool   verbose = false;
};

DemoConfig ParseArgs(int argc, char** argv) {
    DemoConfig c;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--model"    && i+1<argc) c.model_path = argv[++i];
        else if (a == "--problem"  && i+1<argc) c.problem    = argv[++i];
        else if (a == "--dim"      && i+1<argc) c.dim        = std::stoi(argv[++i]);
        else if (a == "--mesh"     && i+1<argc) c.mesh_type  = argv[++i];
        else if (a == "--ref"      && i+1<argc) c.ref        = std::stoi(argv[++i]);
        else if (a == "--order"    && i+1<argc) c.order      = std::stoi(argv[++i]);
        else if (a == "--k"        && i+1<argc) c.k_ratio    = std::stod(argv[++i]);
        else if (a == "--peclet"   && i+1<argc) c.peclet     = std::stod(argv[++i]);
        else if (a == "--contrast" && i+1<argc) c.material_contrast = std::stod(argv[++i]);
        else if (a == "--rtol"     && i+1<argc) c.rtol       = std::stod(argv[++i]);
        else if (a == "--maxit"    && i+1<argc) c.maxit      = std::stoi(argv[++i]);
        else if (a == "--compare") c.compare = true;
        else if (a == "--verbose") c.verbose = true;
    }
    return c;
}

// ──────────────────────────────────────────────────────────────
// 辅助：自定义系数（与 heat_benchmark.cpp 相同）
// ──────────────────────────────────────────────────────────────
class AnisoK : public MatrixCoefficient {
    double kx_, ky_, kz_;
public:
    AnisoK(int d, double kx, double ky, double kz = 1.0)
        : MatrixCoefficient(d), kx_(kx), ky_(ky), kz_(kz) {}
    void Eval(DenseMatrix& K, ElementTransformation&,
              const IntegrationPoint&) override {
        int d = GetHeight(); K.SetSize(d); K = 0.0;
        K(0,0) = kx_;
        if (d > 1) K(1,1) = ky_;
        if (d > 2) K(2,2) = kz_;
    }
};

class HeterogeneousK : public Coefficient {
    double k_bg_, k_inc_; Vector center_; double radius_;
public:
    HeterogeneousK(double bg, double inc, Vector c, double r)
        : k_bg_(bg), k_inc_(inc), center_(std::move(c)), radius_(r) {}
    double Eval(ElementTransformation& T,
                const IntegrationPoint& ip) override {
        Vector x(center_.Size()); T.Transform(ip, x);
        double d2 = 0;
        for (int i = 0; i < x.Size(); ++i)
            d2 += (x(i) - center_(i)) * (x(i) - center_(i));
        return (std::sqrt(d2) < radius_) ? k_inc_ : k_bg_;
    }
};

// ──────────────────────────────────────────────────────────────
// 组装特定问题的 MFEM 系统
// ──────────────────────────────────────────────────────────────
struct Problem {
    std::shared_ptr<Mesh>                 mesh;
    std::shared_ptr<H1_FECollection>      fec;
    std::shared_ptr<FiniteElementSpace>   fes;
    std::shared_ptr<BilinearForm>         a;
    std::shared_ptr<LinearForm>           b;
    std::shared_ptr<GridFunction>         x_gf;
    std::shared_ptr<Coefficient>          k_coef;
    std::shared_ptr<MatrixCoefficient>    K_coef;
    std::shared_ptr<VectorCoefficient>    v_coef;
    std::shared_ptr<Coefficient>          src_coef;
    std::shared_ptr<Coefficient>          robin_coef;
    OperatorPtr     A;
    Vector          B, X;
    HeatProblemDesc desc;
    double          assemble_ms = 0.0;
};

Element::Type ParseMeshType(const std::string& s, int dim) {
    if (dim == 2) {
        if (s == "tri")  return Element::TRIANGLE;
        return Element::QUADRILATERAL;
    } else {
        if (s == "tet")  return Element::TETRAHEDRON;
        return Element::HEXAHEDRON;
    }
}

Problem BuildProblem(const DemoConfig& cfg)
{
    using Clock = std::chrono::high_resolution_clock;
    auto ms_since = [](Clock::time_point t){
        return std::chrono::duration<double, std::milli>(
            Clock::now() - t).count();
    };

    Problem p;

    // 1. 网格
    Element::Type etype = ParseMeshType(cfg.mesh_type, cfg.dim);
    int n0 = (etype == Element::TRIANGLE
              || etype == Element::TETRAHEDRON) ? 2 : 1;
    if (cfg.dim == 2) {
        p.mesh = std::make_shared<Mesh>(
            Mesh::MakeCartesian2D(n0, n0, etype, true, 1.0, 1.0));
    } else {
        p.mesh = std::make_shared<Mesh>(
            Mesh::MakeCartesian3D(n0, n0, n0, etype, 1.0, 1.0, 1.0));
    }
    p.mesh->EnsureNodes();
    for (int i = 0; i < cfg.ref; ++i) p.mesh->UniformRefinement();

    // 2. 有限元空间
    p.fec.reset(new H1_FECollection(cfg.order, cfg.dim));
    p.fes.reset(new FiniteElementSpace(p.mesh.get(), p.fec.get()));

    auto t0 = Clock::now();

    p.a.reset(new BilinearForm(p.fes.get()));
    p.b.reset(new LinearForm(p.fes.get()));
    p.src_coef.reset(new ConstantCoefficient(1.0));
    p.b->AddDomainIntegrator(new DomainLFIntegrator(*p.src_coef));

    // 3. 根据问题类型添加积分器
    if (cfg.problem == "steady") {
        p.desc.problem_family = ProblemFamily::SteadyHeat;
        p.desc.k_ratio = 1.0;
        p.k_coef.reset(new ConstantCoefficient(cfg.k_value));
        p.a->AddDomainIntegrator(new DiffusionIntegrator(*p.k_coef));
    }
    else if (cfg.problem == "aniso") {
        p.desc.problem_family = ProblemFamily::AnisoHeat;
        p.desc.k_ratio = cfg.k_ratio;
        double kx = cfg.k_ratio, ky = 1.0, kz = 1.0;
        p.K_coef.reset(new AnisoK(cfg.dim, kx, ky, kz));
        p.a->AddDomainIntegrator(new DiffusionIntegrator(*p.K_coef));
    }
    else if (cfg.problem == "multimat") {
        p.desc.problem_family    = ProblemFamily::MultiMat;
        p.desc.material_contrast = cfg.material_contrast;
        Vector center(cfg.dim); center = 0.5;
        p.k_coef.reset(new HeterogeneousK(1.0, cfg.material_contrast,
                                            center, 0.25));
        p.a->AddDomainIntegrator(new DiffusionIntegrator(*p.k_coef));
    }
    else if (cfg.problem == "convdiff") {
        p.desc.problem_family = ProblemFamily::ConvDiff;
        p.desc.peclet = cfg.peclet;
        double h = p.mesh->GetElementSize(0);
        if (h < 1e-12) h = 1e-3;
        double v_mag = 1.0;
        double k_val = v_mag * h / (2.0 * std::max(cfg.peclet, 0.01));
        p.k_coef.reset(new ConstantCoefficient(k_val));
        p.a->AddDomainIntegrator(new DiffusionIntegrator(*p.k_coef));

        Vector vv(cfg.dim); vv = 0.0; vv(0) = v_mag;
        if (cfg.dim > 1) vv(1) = 0.5 * v_mag;
        p.v_coef.reset(new VectorConstantCoefficient(vv));
        p.a->AddDomainIntegrator(new ConvectionIntegrator(*p.v_coef, 1.0));
    }
    else if (cfg.problem == "robin") {
        p.desc.problem_family    = ProblemFamily::RobinHeat;
        p.desc.material_contrast = cfg.robin_h;
        p.k_coef.reset(new ConstantCoefficient(1.0));
        p.a->AddDomainIntegrator(new DiffusionIntegrator(*p.k_coef));
        p.robin_coef.reset(new ConstantCoefficient(cfg.robin_h));
        p.a->AddBoundaryIntegrator(new MassIntegrator(*p.robin_coef));
    }
    else {
        std::fprintf(stderr, "[ERR] 未知问题类型: %s\n", cfg.problem.c_str());
        std::exit(1);
    }

    // 4. 组装 + 边界条件
    p.a->Assemble();
    p.a->Finalize(1);            // 排序 CSR
    p.b->Assemble();

    Array<int> ess_tdof_list;
    bool apply_dirichlet = (cfg.problem != "robin");
    if (apply_dirichlet && p.mesh->bdr_attributes.Size() > 0) {
        Array<int> ess_bdr(p.mesh->bdr_attributes.Max());
        ess_bdr = 1;
        p.fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }
    p.x_gf.reset(new GridFunction(p.fes.get()));
    *p.x_gf = 0.0;
    p.a->FormLinearSystem(ess_tdof_list, *p.x_gf, *p.b, p.A, p.X, p.B);

    p.assemble_ms = ms_since(t0);
    return p;
}

// ──────────────────────────────────────────────────────────────
// 主函数
// ──────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    DemoConfig cfg = ParseArgs(argc, argv);

    std::printf("╔══════════════════════════════════════════════════════╗\n");
    std::printf("║   MFEM 热传导 + PyTorch MLP 自动求解器选择           ║\n");
    std::printf("╚══════════════════════════════════════════════════════╝\n");
    std::printf("  问题:    %s\n",          cfg.problem.c_str());
    std::printf("  网格:    %dD %s (ref=%d, P%d)\n",
                cfg.dim, cfg.mesh_type.c_str(), cfg.ref, cfg.order);
    std::printf("  模型:    %s\n",          cfg.model_path.c_str());
    std::printf("  容差:    %.0e\n",         cfg.rtol);

    // 1. 加载模型
    HeatSolverSelector sel(cfg.model_path);
    sel.SetVerbose(cfg.verbose);

    // 2. 构造 MFEM 问题
    std::printf("\n── 组装系统 ──\n");
    auto prob = BuildProblem(cfg);
    SparseMatrix& A = *dynamic_cast<SparseMatrix*>(prob.A.Ptr());

    std::printf("  DOF: %d    elements: %d    组装: %.2f ms\n",
                prob.fes->GetTrueVSize(), prob.mesh->GetNE(),
                prob.assemble_ms);

    // 3. AI 推理
    std::printf("\n── AI 推理 ──\n");
    auto pick = sel.Pick(A, *prob.mesh, *prob.fes, prob.desc);
    pick.mat_features.Print();
    pick.Print();

    // 4. 运行推荐求解器
    std::printf("\n── 运行推荐求解器 ──\n");
    Vector x_sol(prob.B.Size());
    auto res = sel.Solve(pick.best_solver, A, prob.B, x_sol,
                          cfg.rtol, cfg.maxit);
    std::printf("  %s: %s  iters=%d  residual=%.2e  "
                "setup=%.1fms solve=%.1fms total=%.1fms\n",
                res.solver_used.c_str(),
                res.converged ? "\033[32m✓\033[0m" : "\033[31m✗\033[0m",
                res.iterations, res.final_residual,
                res.setup_ms, res.solve_ms, res.total_ms);

    // 5. 可选：对比所有求解器
    if (true) {
        std::printf("\n── 对比所有求解器 (验证 AI 推荐) ──\n");
        std::printf("  %-16s %8s %8s %11s %10s %s\n",
                    "求解器", "迭代", "残差", "耗时(ms)", "vs AI", "状态");
        std::printf("  %s\n", std::string(72, '-').c_str());

        std::vector<HeatSolveResult> results;
        double ai_time = res.total_ms;

        for (const auto& name : sel.ClassNames()) {
            Vector xx(prob.B.Size());
            auto r = sel.Solve(name, A, prob.B, xx, cfg.rtol, cfg.maxit);
            results.push_back(r);

            double ratio = (r.total_ms > 0 && r.converged)
                           ? r.total_ms / std::max(ai_time, 1e-3)
                           : 0.0;
            bool is_ai_pick = (name == pick.best_solver);

            std::printf("  %s%-16s %8d %8.1e %11.2f %9.2fx %s%s\n",
                        is_ai_pick ? "\033[1;33m" : "",
                        name.c_str(),
                        r.iterations,
                        r.final_residual,
                        r.total_ms,
                        ratio,
                        r.converged ? "✓" : "✗",
                        is_ai_pick ? " ← AI\033[0m" : "");
        }

        // 找实际最优
        int actual_best = -1;
        double best_t = 1e30;
        for (size_t i = 0; i < results.size(); ++i) {
            if (results[i].converged && results[i].total_ms < best_t) {
                best_t = results[i].total_ms;
                actual_best = (int)i;
            }
        }
        if (actual_best >= 0) {
            const auto& actual_name = sel.ClassNames()[actual_best];
            bool match = (actual_name == pick.best_solver);
            std::printf("\n  实际最优: \033[36m%s\033[0m (%.1f ms)   "
                        "AI 推荐: %s   %s\n",
                        actual_name.c_str(), best_t,
                        pick.best_solver.c_str(),
                        match ? "\033[32m✓ 匹配\033[0m"
                              : "\033[33m~ 不完全匹配\033[0m");
        }
    }

    std::printf("\n✅ 完成\n");
    return 0;
}