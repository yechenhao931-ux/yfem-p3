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
   int problem_type; // 0 = steady Poisson, 1 = transient implicit step
   double kappa;
   double alpha;
   double dt;
   double aniso; // anisotropy ratio between principal diffusivities
};

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

   BilinearForm K(&fes);
   K.AddDomainIntegrator(new DiffusionIntegrator(kcoef));
   K.Assemble();

   SparseMatrix A;
   Vector B, X;

   if (cfg.problem_type == 0)
   {
      // Steady Poisson: solve K u = f.
      K.FormLinearSystem(ess_tdof_list, u, rhs, A, X, B);
   }
   else
   {
      // Transient implicit step: form T = M + dt*K, then eliminate essential
      // dofs to keep the system SPD. This mimics the matrix that ex16 hands to
      // its CG solver inside ImplicitSolve.
      BilinearForm M(&fes);
      ConstantCoefficient rho(1.0);
      M.AddDomainIntegrator(new MassIntegrator(rho));
      M.Assemble();

      SparseMatrix Kmat, Mmat;
      K.FormSystemMatrix(Array<int>(), Kmat); // no elimination, finalize only
      M.FormSystemMatrix(Array<int>(), Mmat);

      std::unique_ptr<SparseMatrix> T(Add(1.0, Mmat, cfg.dt, Kmat));
      for (int i = 0; i < ess_tdof_list.Size(); ++i)
      {
         T->EliminateRowColDiag(ess_tdof_list[i], 1.0);
      }
      A.Swap(*T);

      B.SetSize(rhs.Size());
      B = rhs;
      for (int i = 0; i < ess_tdof_list.Size(); ++i)
      {
         B(ess_tdof_list[i]) = 0.0;
      }
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

   // Fill matrix-level features.
   FillMatrixFeatures(A, feat_out);

   A_out.Swap(A);
   b_out = B;
}

double TimeSolver(int solver_id, SparseMatrix &A, const Vector &b,
                  int max_iter, double rtol, int repeats)
{
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
   int max_iter = 2000;
   double rtol = 1e-10;
   int repeats = 1;
   int sample_seed = 1234;
   int num_samples = 0; // 0 = enumerate the full grid

   OptionsParser args(argc, argv);
   args.AddOption(&out_path, "-o", "--output", "Output CSV path.");
   args.AddOption(&max_iter, "-mi", "--max-iter", "Max Krylov iterations.");
   args.AddOption(&rtol, "-rt", "--rtol", "Relative residual tolerance.");
   args.AddOption(&repeats, "-rp", "--repeats",
                  "Times each solver is run per problem (averaged).");
   args.AddOption(&sample_seed, "-s", "--seed",
                  "RNG seed for randomized parameter sampling.");
   args.AddOption(&num_samples, "-n", "--num-samples",
                  "If > 0, draw this many random samples instead of "
                  "enumerating the full grid.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(std::cout); return 1; }

   // Pool of meshes that ship with MFEM. We pick a few of varying topology
   // and dimension so the model sees representative connectivity patterns.
   const std::vector<std::string> meshes = {
      "data/inline-tri.mesh",
      "data/inline-quad.mesh",
      "data/star.mesh",
      "data/disc-nurbs.mesh",
      "data/inline-hex.mesh",
      "data/inline-tet.mesh",
      "data/fichera.mesh"
   };
   const std::vector<int>    ref_levels_grid = {1, 2, 3};
   const std::vector<int>    order_grid      = {1, 2, 3};
   const std::vector<int>    type_grid       = {0, 1};
   const std::vector<double> kappa_grid      = {0.1, 1.0, 10.0};
   const std::vector<double> dt_grid         = {1e-3, 1e-2, 1e-1};
   const std::vector<double> aniso_grid      = {1.0, 10.0, 100.0};

   // Build the full configuration list.
   std::vector<ProblemConfig> configs;
   for (const auto &m : meshes)
   for (int rl : ref_levels_grid)
   for (int o  : order_grid)
   for (int pt : type_grid)
   for (double k : kappa_grid)
   for (double an : aniso_grid)
   {
      ProblemConfig c;
      c.mesh_file = m;
      c.ref_levels = rl;
      c.order = o;
      c.problem_type = pt;
      c.kappa = k;
      c.alpha = 0.0;
      c.dt = (pt == 1) ? dt_grid[1] : 0.0; // single representative dt
      c.aniso = an;
      configs.push_back(c);

      if (pt == 1)
      {
         for (double dt : dt_grid)
         {
            if (dt == dt_grid[1]) { continue; }
            ProblemConfig c2 = c;
            c2.dt = dt;
            configs.push_back(c2);
         }
      }
   }

   if (num_samples > 0 && num_samples < (int)configs.size())
   {
      std::mt19937 rng(sample_seed);
      std::shuffle(configs.begin(), configs.end(), rng);
      configs.resize(num_samples);
   }

   std::ofstream csv(out_path);
   if (!csv) { std::cerr << "Cannot open " << out_path << "\n"; return 2; }
   WriteCSVHeader(csv, SOLVER_COUNT);

   std::cout << "Collecting " << configs.size() << " problem instances...\n";
   int written = 0;
   for (size_t i = 0; i < configs.size(); ++i)
   {
      const ProblemConfig &cfg = configs[i];
      try
      {
         SparseMatrix A;
         Vector b;
         FeatureVector feat;
         AssembleHeatSystem(cfg, A, b, feat);

         std::vector<double> times(SOLVER_COUNT, 0.0);
         for (int s = 0; s < SOLVER_COUNT; ++s)
         {
            times[s] = TimeSolver(s, A, b, max_iter, rtol, repeats);
         }

         WriteCSVRow(csv, feat, times);
         csv.flush();
         ++written;

         std::cout << "[" << i + 1 << "/" << configs.size() << "] mesh="
                   << cfg.mesh_file << " r=" << cfg.ref_levels
                   << " o=" << cfg.order << " type=" << cfg.problem_type
                   << " n=" << static_cast<int>(feat[F_N])
                   << " best=" << SolverName(static_cast<int>(
                         std::min_element(times.begin(), times.end())
                         - times.begin()))
                   << "\n";
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

   std::cout << "Wrote " << written << " rows to " << out_path << "\n";
   return 0;
}
