// AI Solver Selector - end-to-end demo.
//
// Re-uses the steady-Poisson assembly from collect_heat_data, asks the trained
// TorchScript model to pick a solver for the resulting linear system, and
// times the solve against the ground-truth optimum chosen by exhaustive
// trial. This demonstrates how to call SolverSelectorModel from a real MFEM
// application.
//
// Run:
//   ./ai_solver_demo -m ai_solver_selector/inference/solver_selector.ts \
//                    -mesh data/star.mesh -r 3 -o 2

#include "mfem.hpp"
#include "solver_selector.hpp"

#include <iostream>

using namespace mfem;
using namespace ai_ss;

int main(int argc, char *argv[])
{
   const char *model_path = "ai_solver_selector/inference/solver_selector.ts";
   const char *mesh_file = "data/star.mesh";
   int ref_levels = 3;
   int order = 2;
   double kappa = 1.0;
   double aniso = 1.0;
   int max_iter = 2000;
   double rtol = 1e-10;

   OptionsParser args(argc, argv);
   args.AddOption(&model_path, "-m", "--model", "TorchScript model path.");
   args.AddOption(&mesh_file, "-mesh", "--mesh", "Mesh file.");
   args.AddOption(&ref_levels, "-r", "--refine", "Uniform refinement levels.");
   args.AddOption(&order, "-o", "--order", "FE polynomial order.");
   args.AddOption(&kappa, "-k", "--kappa", "Diffusivity coefficient.");
   args.AddOption(&aniso, "-an", "--aniso", "Anisotropy ratio.");
   args.AddOption(&max_iter, "-mi", "--max-iter", "Max Krylov iterations.");
   args.AddOption(&rtol, "-rt", "--rtol", "Relative tolerance.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(std::cout); return 1; }

   // Assemble a steady Poisson system. Same recipe as collect_heat_data so
   // features stay consistent.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();
   for (int i = 0; i < ref_levels; ++i) { mesh.UniformRefinement(); }

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);

   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   ConstantCoefficient one(1.0);
   LinearForm rhs(&fes);
   rhs.AddDomainIntegrator(new DomainLFIntegrator(one));
   rhs.Assemble();

   GridFunction u(&fes); u = 0.0;

   DenseMatrix Kmat(dim);
   Kmat = 0.0;
   Kmat(0, 0) = kappa;
   if (dim > 1) { Kmat(1, 1) = kappa * aniso; }
   if (dim > 2) { Kmat(2, 2) = kappa * aniso; }
   MatrixConstantCoefficient kcoef(Kmat);

   BilinearForm K(&fes);
   K.AddDomainIntegrator(new DiffusionIntegrator(kcoef));
   K.Assemble();

   SparseMatrix A;
   Vector B, X;
   K.FormLinearSystem(ess_tdof_list, u, rhs, A, X, B);

   // Predict.
   SolverSelectorModel model(model_path);
   FeatureVector feat = ExtractFeatures(/*problem_type=*/0, dim, order,
                                        ref_levels, kappa, /*alpha=*/0.0,
                                        /*dt=*/0.0, aniso, A);
   const auto pred_times = model.PredictTimes(feat);
   const int picked = std::min_element(pred_times.begin(), pred_times.end()) -
                      pred_times.begin();

   std::cout << "n=" << A.Height() << "  nnz=" << A.NumNonZeroElems() << "\n";
   std::cout << "Predicted runtimes (s):\n";
   for (int s = 0; s < SOLVER_COUNT; ++s)
   {
      std::cout << "  [" << s << "] " << SolverName(s) << ": "
                << pred_times[s] << (s == picked ? "  <-- picked\n" : "\n");
   }

   // Run the picked solver.
   {
      SolverPair sp = MakeSolver(picked, A, max_iter, rtol);
      Vector x(B.Size()); x = 0.0;
      StopWatch sw; sw.Start();
      sp.solver->Mult(B, x);
      sw.Stop();
      std::cout << "AI choice (" << SolverName(picked) << ") solved in "
                << sw.RealTime() << "s, iters="
                << sp.solver->GetNumIterations()
                << ", converged=" << sp.solver->GetConverged() << "\n";
   }

   // Ground-truth: try every solver and report the actual optimum.
   double best_time = std::numeric_limits<double>::infinity();
   int best_id = -1;
   for (int s = 0; s < SOLVER_COUNT; ++s)
   {
      SolverPair sp = MakeSolver(s, A, max_iter, rtol);
      Vector x(B.Size()); x = 0.0;
      StopWatch sw; sw.Start();
      sp.solver->Mult(B, x);
      sw.Stop();
      const double t =
         sp.solver->GetConverged() ? sw.RealTime() : sw.RealTime() * 10.0 + 1.0;
      std::cout << "  truth [" << s << "] " << SolverName(s) << ": " << t << "s\n";
      if (t < best_time) { best_time = t; best_id = s; }
   }

   std::cout << "Optimal solver was " << SolverName(best_id)
             << " (id=" << best_id << ", " << best_time << "s).\n";
   std::cout << "AI matched optimum: " << (picked == best_id ? "yes" : "no")
             << "\n";

   return 0;
}
