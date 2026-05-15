// AI Solver Selector - C++ inference API.
//
// Loads a TorchScript model produced by training/train.py and exposes:
//
//   1. ExtractFeatures(...) - build the feature vector for a heat-conduction
//      problem from its (mesh + assembled SparseMatrix + scalar parameters).
//   2. PredictBestSolver(features) - run the model and return the SolverId
//      with the smallest predicted runtime.
//   3. MakeBestSolver(A, ...) - convenience: assemble + predict + construct
//      the chosen IterativeSolver / preconditioner pair, ready to .Mult().
//
// This header is libtorch-aware. Including it from MFEM application code
// requires linking against libtorch (see CMakeLists.txt in this directory).

#ifndef AI_SS_INFERENCE_HPP
#define AI_SS_INFERENCE_HPP

#include "mfem.hpp"
#include "../data_collector/matrix_features.hpp"
#include "../data_collector/solver_registry.hpp"

#include <torch/script.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace ai_ss
{

class SolverSelectorModel
{
public:
   explicit SolverSelectorModel(const std::string &torchscript_path)
   {
      try
      {
         module_ = torch::jit::load(torchscript_path);
         module_.eval();
      }
      catch (const c10::Error &e)
      {
         throw std::runtime_error(
            "Failed to load TorchScript model from " + torchscript_path +
            ": " + e.what());
      }
   }

   // Run the network on a single feature vector and return the predicted
   // runtime per solver (in seconds). The vector has SOLVER_COUNT entries.
   std::vector<double> PredictTimes(const FeatureVector &features) const
   {
      torch::NoGradGuard g;
      auto x = torch::empty({1, F_COUNT}, torch::kFloat32);
      auto x_acc = x.accessor<float, 2>();
      for (int i = 0; i < F_COUNT; ++i)
      {
         x_acc[0][i] = static_cast<float>(features[i]);
      }
      std::vector<torch::jit::IValue> inputs;
      inputs.emplace_back(x);

      at::Tensor out = module_.forward(inputs).toTensor();
      // Model emits log-time; exponentiate back into seconds.
      out = out.exp();

      std::vector<double> times(SOLVER_COUNT, 0.0);
      auto out_acc = out.accessor<float, 2>();
      for (int s = 0; s < SOLVER_COUNT; ++s)
      {
         times[s] = static_cast<double>(out_acc[0][s]);
      }
      return times;
   }

   int PredictBestSolver(const FeatureVector &features) const
   {
      const auto times = PredictTimes(features);
      return static_cast<int>(
         std::min_element(times.begin(), times.end()) - times.begin());
   }

   // Monte Carlo dropout: average n_samples stochastic forward passes,
   // then report (mean_times, max_log_std) over the predicted log runtimes.
   // Pass these to PredictWithFallback() to fall back to a safe default
   // when the model is uncertain.
   struct UncertaintyReport
   {
      std::vector<double> mean_times;  // size SOLVER_COUNT
      std::vector<double> log_std;     // size SOLVER_COUNT
   };

   UncertaintyReport PredictWithMCDropout(const FeatureVector &features,
                                          int n_samples = 32) const
   {
      torch::NoGradGuard g;
      // Train mode so dropout layers stay stochastic.
      module_.train();

      auto x = torch::empty({1, F_COUNT}, torch::kFloat32);
      auto x_acc = x.accessor<float, 2>();
      for (int i = 0; i < F_COUNT; ++i)
      {
         x_acc[0][i] = static_cast<float>(features[i]);
      }
      std::vector<torch::jit::IValue> inputs;
      inputs.emplace_back(x);

      std::vector<std::vector<double>> samples(n_samples,
         std::vector<double>(SOLVER_COUNT, 0.0));
      for (int k = 0; k < n_samples; ++k)
      {
         at::Tensor out = module_.forward(inputs).toTensor();
         auto out_acc = out.accessor<float, 2>();
         for (int s = 0; s < SOLVER_COUNT; ++s)
         {
            samples[k][s] = static_cast<double>(out_acc[0][s]);
         }
      }

      module_.eval();

      UncertaintyReport rep;
      rep.mean_times.assign(SOLVER_COUNT, 0.0);
      rep.log_std.assign(SOLVER_COUNT, 0.0);
      for (int s = 0; s < SOLVER_COUNT; ++s)
      {
         double mean = 0.0;
         for (int k = 0; k < n_samples; ++k) { mean += samples[k][s]; }
         mean /= n_samples;
         double var = 0.0;
         for (int k = 0; k < n_samples; ++k)
         {
            const double d = samples[k][s] - mean;
            var += d * d;
         }
         var /= std::max(1, n_samples - 1);
         rep.mean_times[s] = std::exp(mean);
         rep.log_std[s] = std::sqrt(var);
      }
      return rep;
   }

   // Production wrapper. Returns the argmin pick under the mean
   // prediction; if MC-dropout uncertainty exceeds ``unc_threshold`` on
   // any solver, falls back to ``safe_default`` (typically CG+Jacobi).
   int PredictBestSolverSafe(const FeatureVector &features,
                             int safe_default = SOLVER_CG_JACOBI,
                             double unc_threshold = 1.5,
                             int n_samples = 16) const
   {
      const auto rep = PredictWithMCDropout(features, n_samples);
      const double max_std =
         *std::max_element(rep.log_std.begin(), rep.log_std.end());
      if (max_std > unc_threshold) { return safe_default; }
      return static_cast<int>(
         std::min_element(rep.mean_times.begin(), rep.mean_times.end())
         - rep.mean_times.begin());
   }

private:
   mutable torch::jit::script::Module module_;
};

// Build the feature vector for a problem instance the same way
// collect_heat_data does at training time. Pass the scalar parameters that
// describe the problem (problem_type, dim, order, ref_levels, kappa, alpha,
// dt, aniso, reaction, velocity) and the assembled, post-elimination
// SparseMatrix. The reaction/velocity arguments default to 0 so existing
// callers keep working unchanged.
inline FeatureVector ExtractFeatures(int problem_type, int dim, int order,
                                     int ref_levels, double kappa,
                                     double alpha, double dt, double aniso,
                                     const mfem::SparseMatrix &A,
                                     double reaction = 0.0,
                                     double velocity = 0.0)
{
   FeatureVector f;
   f.fill(0.0);
   f[F_PROBLEM_TYPE] = problem_type;
   f[F_DIM] = dim;
   f[F_ORDER] = order;
   f[F_REF_LEVELS] = ref_levels;
   f[F_KAPPA] = kappa;
   f[F_ALPHA] = alpha;
   f[F_DT] = dt;
   f[F_ANISO] = aniso;
   f[F_REACTION] = reaction;
   f[F_VELOCITY] = velocity;
   FillMatrixFeatures(A, f);
   return f;
}

// Convenience: predict + build. Returns the (solver, prec) pair already wired
// to A; just call solver->Mult(b, x) on it.
inline SolverPair MakeBestSolver(const SolverSelectorModel &model,
                                 const FeatureVector &features,
                                 mfem::SparseMatrix &A,
                                 int max_iter = 2000,
                                 double rtol = 1e-10,
                                 int *out_solver_id = nullptr)
{
   const int sid = model.PredictBestSolver(features);
   if (out_solver_id) { *out_solver_id = sid; }
   return MakeSolver(sid, A, max_iter, rtol);
}

} // namespace ai_ss

#endif // AI_SS_INFERENCE_HPP
