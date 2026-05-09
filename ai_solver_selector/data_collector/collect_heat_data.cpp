// AI Solver Selector - heat-conduction data collector
//
// Sweeps across heat-conduction FEM problem configurations (steady Poisson and
// transient implicit-step systems based on examples ex1 / ex16) and times a
// fixed list of (Krylov solver + preconditioner) pairs on each one. Writes a
// CSV with one row per problem instance:
//
//   <feature_0>,...,<feature_F-1>,t_solver_0,...,t_solver_S-1
//
// The feature layout is defined in matrix_features.hpp; the solver layout in
// solver_registry.hpp. Both must stay in sync with the PyTorch training side.
//
// Build (from the MFEM root):
//   make ai_collect_heat_data
//
// Sample run:
//   ./ai_collect_heat_data -o data/heat_solver_dataset.csv

#include "mfem.hpp"
#include "matrix_features.hpp"
#include "solver_registry.hpp"

#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace mfem;
using namespace ai_ss;

namespace
{

struct ProblemConfig
{
   std::string mesh_file;
   int ref_levels;
   int order;
   int problem_type; // 0=steady Poisson, 1=transient implicit, 2=mass,
                     // 3=reaction-diffusion, 4=convection-diffusion
   double kappa;
   double alpha;
   double dt;
   double aniso;     // diffusivity anisotropy between principal axes
   double reaction;  // reaction coefficient r in (K + r*M)
   double velocity;  // convection-velocity magnitude |beta|
};

const char *ProblemTypeName(int pt)
{
   switch (pt)
   {
      case 0: return "Poisson";
      case 1: return "Transient(M+dt*K)";
      case 2: return "Mass";
      case 3: return "Reaction-Diffusion";
      case 4: return "Convection-Diffusion";
      default: return "?";
   }
}

// Anisotropic diagonal diffusion matrix coefficient kappa * diag(1, aniso, aniso).
class AnisoDiffusionCoefficient : public MatrixCoefficient
{
public:
   AnisoDiffusionCoefficient(int dim, double kappa, double aniso)
      : MatrixCoefficient(dim), kappa_(kappa), aniso_(aniso) {}

   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      K.SetSize(width);
      K = 0.0;
      K(0, 0) = kappa_;
      if (width > 1) { K(1, 1) = kappa_ * aniso_; }
      if (width > 2) { K(2, 2) = kappa_ * aniso_; }
   }

private:
   double kappa_;
   double aniso_;
};

// Assemble the linear system for the requested problem type. Returns the
// (sparse) matrix A and the RHS vector b. The matrix is always SPD.
void AssembleHeatSystem(const ProblemConfig &cfg, SparseMatrix &A_out,
                        Vector &b_out, FeatureVector &feat_out)
{
   Mesh mesh(cfg.mesh_file.c_str(), 1, 1);
   const int dim = mesh.Dimension();
   for (int i = 0; i < cfg.ref_levels; ++i) { mesh.UniformRefinement(); }

   H1_FECollection fec(cfg.order, dim);
   FiniteElementSpace fes(&mesh, &fec);

   // Essential dofs: tag the entire boundary.
   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // RHS: f = 1 -> simple thermal source.
   ConstantCoefficient one(1.0);
   LinearForm rhs(&fes);
   rhs.AddDomainIntegrator(new DomainLFIntegrator(one));
   rhs.Assemble();

   GridFunction u(&fes);
   u = 0.0;

   AnisoDiffusionCoefficient kcoef(dim, cfg.kappa, cfg.aniso);
   ConstantCoefficient unit_rho(1.0);

   SparseMatrix A;
   Vector B, X;

   auto eliminate_and_zero_rhs = [&](SparseMatrix &S, Vector &b)
   {
      for (int i = 0; i < ess_tdof_list.Size(); ++i)
      {
         S.EliminateRowColDiag(ess_tdof_list[i], 1.0);
         b(ess_tdof_list[i]) = 0.0;
      }
   };

   if (cfg.problem_type == 0)
   {
      // Steady Poisson: solve K u = f.
      BilinearForm K(&fes);
      K.AddDomainIntegrator(new DiffusionIntegrator(kcoef));
      K.Assemble();
      K.FormLinearSystem(ess_tdof_list, u, rhs, A, X, B);
   }
   else if (cfg.problem_type == 1)
   {
      // Transient implicit step: form T = M + dt*K, then eliminate essential
      // dofs to keep the system SPD (mirrors ex16's ImplicitSolve matrix).
      BilinearForm K(&fes);
      K.AddDomainIntegrator(new DiffusionIntegrator(kcoef));
      K.Assemble();

      BilinearForm M(&fes);
      M.AddDomainIntegrator(new MassIntegrator(unit_rho));
      M.Assemble();

      SparseMatrix Kmat, Mmat;
      K.FormSystemMatrix(Array<int>(), Kmat);
      M.FormSystemMatrix(Array<int>(), Mmat);

      std::unique_ptr<SparseMatrix> T(Add(1.0, Mmat, cfg.dt, Kmat));
      A.Swap(*T);
      B.SetSize(rhs.Size());
      B = rhs;
      eliminate_and_zero_rhs(A, B);
   }
   else if (cfg.problem_type == 2)
   {
      // Pure mass solve: M u = f. Trivial case but still useful for the model
      // because mass matrices are well-conditioned and Jacobi/CG dominate.
      BilinearForm M(&fes);
      M.AddDomainIntegrator(new MassIntegrator(unit_rho));
      M.Assemble();
      M.FormLinearSystem(ess_tdof_list, u, rhs, A, X, B);
   }
   else if (cfg.problem_type == 3)
   {
      // Reaction-diffusion: (K + r*M) u = f. SPD; conditioning depends on
      // r relative to kappa.
      BilinearForm K(&fes);
      K.AddDomainIntegrator(new DiffusionIntegrator(kcoef));
      K.Assemble();

      BilinearForm M(&fes);
      M.AddDomainIntegrator(new MassIntegrator(unit_rho));
      M.Assemble();

      SparseMatrix Kmat, Mmat;
      K.FormSystemMatrix(Array<int>(), Kmat);
      M.FormSystemMatrix(Array<int>(), Mmat);

      std::unique_ptr<SparseMatrix> S(Add(1.0, Kmat, cfg.reaction, Mmat));
      A.Swap(*S);
      B.SetSize(rhs.Size());
      B = rhs;
      eliminate_and_zero_rhs(A, B);
   }
   else if (cfg.problem_type == 4)
   {
      // Convection-diffusion: (K + ConvectionIntegrator(beta)) u = f. The
      // convection term is non-symmetric, so this gives the model examples
      // where CG won't converge and GMRES/BiCGStab should win.
      Vector beta_vals(dim);
      beta_vals = 0.0;
      beta_vals(0) = cfg.velocity;
      if (dim > 1) { beta_vals(1) = 0.5 * cfg.velocity; }
      VectorConstantCoefficient beta(beta_vals);

      BilinearForm L(&fes);
      L.AddDomainIntegrator(new DiffusionIntegrator(kcoef));
      L.AddDomainIntegrator(new ConvectionIntegrator(beta));
      L.Assemble();
      L.FormLinearSystem(ess_tdof_list, u, rhs, A, X, B);
   }
   else
   {
      MFEM_ABORT("Unknown problem_type=" << cfg.problem_type);
   }

   // Fill problem-level features.
   feat_out.fill(0.0);
   feat_out[F_PROBLEM_TYPE] = static_cast<double>(cfg.problem_type);
   feat_out[F_DIM] = static_cast<double>(dim);
   feat_out[F_ORDER] = static_cast<double>(cfg.order);
   feat_out[F_REF_LEVELS] = static_cast<double>(cfg.ref_levels);
   feat_out[F_KAPPA] = cfg.kappa;
   feat_out[F_ALPHA] = cfg.alpha;
   feat_out[F_DT] = cfg.dt;
   feat_out[F_ANISO] = cfg.aniso;
   feat_out[F_REACTION] = cfg.reaction;
   feat_out[F_VELOCITY] = cfg.velocity;

   // Fill matrix-level features.
   FillMatrixFeatures(A, feat_out);

   A_out.Swap(A);
   b_out = B;
}

double TimeSolver(int solver_id, SparseMatrix &A, const Vector &b,
                  int max_iter, double rtol, int repeats, bool warmup)
{
   // Optional warm-up: a single untimed solve to warm caches and let the
   // preconditioner allocate its workspace. Greatly reduces the first-run
   // bias for small problems.
   if (warmup)
   {
      SolverPair sp = MakeSolver(solver_id, A, max_iter, rtol);
      Vector x(b.Size());
      x = 0.0;
      sp.solver->Mult(b, x);
   }

   double total = 0.0;
   for (int r = 0; r < repeats; ++r)
   {
      SolverPair sp = MakeSolver(solver_id, A, max_iter, rtol);
      Vector x(b.Size());
      x = 0.0;
      StopWatch sw;
      sw.Start();
      sp.solver->Mult(b, x);
      sw.Stop();
      const double t = sw.RealTime();
      // If a solver diverged, charge a penalty proportional to wall time so
      // that a non-converging combination is never selected as best.
      if (!sp.solver->GetConverged()) { total += t * 10.0 + 1.0; }
      else                            { total += t; }
   }
   return total / repeats;
}

} // namespace

int main(int argc, char *argv[])
{
   const char *out_path = "heat_solver_dataset.csv";
   const char *types_csv = "0,1,2,3,4";
   int max_iter = 2000;
   double rtol = 1e-10;
   int repeats = 2;
   int sample_seed = 1234;
   int num_samples = 0; // 0 = enumerate the full grid
   bool warmup = true;
   bool quick = false;
   bool append = false;
   int max_dofs = 0; // 0 = no cap

   OptionsParser args(argc, argv);
   args.AddOption(&out_path, "-o", "--output", "Output CSV path.");
   args.AddOption(&types_csv, "-pt", "--problem-types",
                  "Comma-separated list of problem_type ids to include "
                  "(0=Poisson, 1=Transient, 2=Mass, 3=ReactDiff, 4=ConvDiff).");
   args.AddOption(&max_iter, "-mi", "--max-iter", "Max Krylov iterations.");
   args.AddOption(&rtol, "-rt", "--rtol", "Relative residual tolerance.");
   args.AddOption(&repeats, "-rp", "--repeats",
                  "Times each solver is run per problem (averaged).");
   args.AddOption(&sample_seed, "-s", "--seed",
                  "RNG seed for randomized parameter sampling.");
   args.AddOption(&num_samples, "-n", "--num-samples",
                  "If > 0, draw this many random samples instead of "
                  "enumerating the full grid.");
   args.AddOption(&warmup, "-w", "--warmup", "-no-w", "--no-warmup",
                  "Run one untimed solve before timed ones.");
   args.AddOption(&quick, "-q", "--quick", "-no-q", "--no-quick",
                  "Use a much smaller grid for fast iteration.");
   args.AddOption(&append, "-a", "--append", "-no-a", "--no-append",
                  "Append to an existing CSV instead of overwriting it.");
   args.AddOption(&max_dofs, "-mx", "--max-dofs",
                  "Skip configurations whose mesh+order would exceed this "
                  "DOF budget (0 = no cap).");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(std::cout); return 1; }

   // Pool of meshes that ship with MFEM. We pick a few of varying topology
   // and dimension so the model sees representative connectivity patterns.
   const std::vector<std::string> meshes = quick
      ? std::vector<std::string>{"data/inline-quad.mesh", "data/star.mesh",
                                  "data/inline-hex.mesh"}
      : std::vector<std::string>{
           "data/inline-tri.mesh",
           "data/inline-quad.mesh",
           "data/star.mesh",
           "data/star-mixed.mesh",
           "data/disc-nurbs.mesh",
           "data/amr-quad.mesh",
           "data/inline-hex.mesh",
           "data/inline-tet.mesh",
           "data/inline-wedge.mesh",
           "data/fichera.mesh",
           "data/beam-tet.mesh",
           "data/beam-hex.mesh"};

   const std::vector<int> ref_levels_grid =
      quick ? std::vector<int>{1, 2} : std::vector<int>{1, 2, 3, 4};
   const std::vector<int> order_grid =
      quick ? std::vector<int>{1, 2} : std::vector<int>{1, 2, 3, 4};
   const std::vector<double> kappa_grid =
      quick ? std::vector<double>{1.0}
            : std::vector<double>{0.01, 0.1, 1.0, 10.0, 100.0};
   const std::vector<double> dt_grid =
      quick ? std::vector<double>{1e-2}
            : std::vector<double>{1e-4, 1e-3, 1e-2, 1e-1, 1.0};
   const std::vector<double> aniso_grid =
      quick ? std::vector<double>{1.0, 10.0}
            : std::vector<double>{1.0, 5.0, 25.0, 100.0, 1000.0};
   const std::vector<double> reaction_grid =
      quick ? std::vector<double>{1.0}
            : std::vector<double>{0.01, 1.0, 100.0};
   const std::vector<double> velocity_grid =
      quick ? std::vector<double>{1.0}
            : std::vector<double>{0.1, 1.0, 10.0, 100.0};

   // Parse the requested problem-type ids.
   std::vector<int> active_types;
   {
      std::string s(types_csv);
      size_t p = 0;
      while (p < s.size())
      {
         size_t q = s.find(',', p);
         if (q == std::string::npos) { q = s.size(); }
         active_types.push_back(std::atoi(s.substr(p, q - p).c_str()));
         p = q + 1;
      }
   }
   auto type_active = [&](int t)
   {
      return std::find(active_types.begin(), active_types.end(), t)
             != active_types.end();
   };

   // Build the full configuration list. Each problem type cycles only the
   // axes that affect its assembly so we don't multiply unrelated knobs.
   std::vector<ProblemConfig> configs;
   auto base = [&](const std::string &m, int rl, int o, int pt, double k,
                   double an)
   {
      ProblemConfig c;
      c.mesh_file = m;
      c.ref_levels = rl;
      c.order = o;
      c.problem_type = pt;
      c.kappa = k;
      c.alpha = 0.0;
      c.dt = 0.0;
      c.aniso = an;
      c.reaction = 0.0;
      c.velocity = 0.0;
      return c;
   };

   for (const auto &m : meshes)
   for (int rl : ref_levels_grid)
   for (int o  : order_grid)
   for (double k  : kappa_grid)
   for (double an : aniso_grid)
   {
      if (type_active(0))
      {
         configs.push_back(base(m, rl, o, 0, k, an));
      }
      if (type_active(1))
      {
         for (double dt : dt_grid)
         {
            ProblemConfig c = base(m, rl, o, 1, k, an);
            c.dt = dt;
            configs.push_back(c);
         }
      }
      if (type_active(2) && an == aniso_grid.front())
      {
         // Mass solve doesn't depend on kappa/aniso; only push one copy per
         // (mesh, refine, order).
         if (k == kappa_grid.front())
         {
            configs.push_back(base(m, rl, o, 2, k, an));
         }
      }
      if (type_active(3))
      {
         for (double r : reaction_grid)
         {
            ProblemConfig c = base(m, rl, o, 3, k, an);
            c.reaction = r;
            configs.push_back(c);
         }
      }
      if (type_active(4))
      {
         for (double v : velocity_grid)
         {
            ProblemConfig c = base(m, rl, o, 4, k, an);
            c.velocity = v;
            configs.push_back(c);
         }
      }
   }

   std::mt19937 rng(sample_seed);
   std::shuffle(configs.begin(), configs.end(), rng);
   if (num_samples > 0 && num_samples < (int)configs.size())
   {
      configs.resize(num_samples);
   }

   const bool file_exists =
      std::ifstream(out_path).good();
   std::ofstream csv(out_path,
                     append ? std::ios::app : std::ios::trunc);
   if (!csv) { std::cerr << "Cannot open " << out_path << "\n"; return 2; }
   if (!append || !file_exists) { WriteCSVHeader(csv, SOLVER_COUNT); }

   std::cout << "Collecting " << configs.size()
             << " problem instances ("
             << (warmup ? "warmup on" : "warmup off")
             << ", repeats=" << repeats
             << (append ? ", append" : ", overwrite")
             << (max_dofs > 0 ? ", max_dofs=" + std::to_string(max_dofs) : "")
             << ")...\n";

   int written = 0;
   int skipped_too_big = 0;
   for (size_t i = 0; i < configs.size(); ++i)
   {
      const ProblemConfig &cfg = configs[i];
      try
      {
         SparseMatrix A;
         Vector b;
         FeatureVector feat;
         AssembleHeatSystem(cfg, A, b, feat);

         if (max_dofs > 0 && feat[F_N] > max_dofs)
         {
            ++skipped_too_big;
            continue;
         }

         std::vector<double> times(SOLVER_COUNT, 0.0);
         for (int s = 0; s < SOLVER_COUNT; ++s)
         {
            times[s] = TimeSolver(s, A, b, max_iter, rtol, repeats, warmup);
         }

         WriteCSVRow(csv, feat, times);
         csv.flush();
         ++written;

         const int best = std::min_element(times.begin(), times.end())
                          - times.begin();
         std::cout << "[" << i + 1 << "/" << configs.size() << "] "
                   << ProblemTypeName(cfg.problem_type)
                   << " mesh=" << cfg.mesh_file
                   << " r=" << cfg.ref_levels << " o=" << cfg.order
                   << " n=" << static_cast<int>(feat[F_N])
                   << " nnz=" << static_cast<int>(feat[F_NNZ])
                   << " sym=" << feat[F_SYMMETRY]
                   << " best=" << SolverName(best)
                   << " (" << times[best] << "s)\n";
      }
      catch (const std::exception &e)
      {
         std::cerr << "  skipped (" << e.what() << ")\n";
      }
      catch (...)
      {
         std::cerr << "  skipped (unknown error)\n";
      }
   }

   std::cout << "Wrote " << written << " rows to " << out_path
             << " (skipped " << skipped_too_big << " over max-dofs)\n";
   return 0;
}
