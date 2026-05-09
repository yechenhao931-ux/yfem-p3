// AI Solver Selector - Matrix feature extraction
//
// Extracts a fixed-length numerical feature vector from a problem instance
// (mesh + assembled SparseMatrix). The same feature layout is used both at
// training-data collection time and at C++ inference time, so a TorchScript
// model trained on these features can be served from C++.

#ifndef AI_SS_MATRIX_FEATURES_HPP
#define AI_SS_MATRIX_FEATURES_HPP

#include "mfem.hpp"
#include <array>
#include <cmath>
#include <vector>

namespace ai_ss
{

// Layout of the feature vector. Keep additions at the end so trained models
// stay backward compatible.
enum FeatureIndex
{
   F_PROBLEM_TYPE = 0, // 0 = steady Poisson, 1 = transient implicit step T=M+dt*K
   F_DIM,              // mesh dimension (2 or 3)
   F_ORDER,            // FE polynomial order
   F_REF_LEVELS,       // refinements applied
   F_N,                // matrix size (DOFs)
   F_NNZ,              // total non-zeros
   F_AVG_NNZ,          // avg non-zeros per row
   F_MAX_NNZ,          // max non-zeros in any row
   F_DENSITY,          // nnz / n^2
   F_DIAG_DOMINANCE,   // mean of |a_ii| / sum_{j!=i} |a_ij|
   F_SYMMETRY,         // ||A - A^T||_F / ||A||_F  (0 means symmetric)
   F_FROB_NORM,        // ||A||_F
   F_COND_EST,         // ratio max(|diag|) / min(|diag|), a cheap proxy
   F_KAPPA,            // diffusivity coefficient
   F_ALPHA,            // nonlinear/anisotropy coefficient
   F_DT,               // time step (0 for steady)
   F_ANISO,            // diffusivity anisotropy (max/min eigenvalue ratio)
   F_COUNT
};

constexpr const char *kFeatureNames[F_COUNT] = {
   "problem_type", "dim",        "order",   "ref_levels", "n",
   "nnz",          "avg_nnz",    "max_nnz", "density",    "diag_dom",
   "symmetry",     "frob_norm",  "cond_est", "kappa",
   "alpha",        "dt",         "aniso"
};

using FeatureVector = std::array<double, F_COUNT>;

// Compute matrix-only structural/numerical features from a SparseMatrix.
// Fills the relevant slots of `out` and leaves problem-type metadata to the
// caller (problem_type, dim, order, ref_levels, kappa, alpha, dt, aniso).
inline void FillMatrixFeatures(const mfem::SparseMatrix &A, FeatureVector &out)
{
   const int n = A.Height();
   const int *I = A.GetI();
   const int *J = A.GetJ();
   const double *V = A.GetData();
   const int nnz = (n > 0) ? I[n] : 0;

   out[F_N] = static_cast<double>(n);
   out[F_NNZ] = static_cast<double>(nnz);
   out[F_AVG_NNZ] = (n > 0) ? static_cast<double>(nnz) / n : 0.0;
   out[F_DENSITY] = (n > 0) ? static_cast<double>(nnz) / (1.0 * n * n) : 0.0;

   // Per-row statistics.
   int max_row_nnz = 0;
   double frob_sq = 0.0;
   double diag_dom_sum = 0.0;
   int diag_dom_count = 0;
   double max_abs_diag = 0.0;
   double min_abs_diag = std::numeric_limits<double>::infinity();

   for (int i = 0; i < n; ++i)
   {
      const int row_nnz = I[i + 1] - I[i];
      if (row_nnz > max_row_nnz) { max_row_nnz = row_nnz; }

      double diag_abs = 0.0;
      double off_abs_sum = 0.0;
      for (int k = I[i]; k < I[i + 1]; ++k)
      {
         const double a = V[k];
         frob_sq += a * a;
         if (J[k] == i) { diag_abs = std::fabs(a); }
         else           { off_abs_sum += std::fabs(a); }
      }
      if (diag_abs > 0.0)
      {
         max_abs_diag = std::max(max_abs_diag, diag_abs);
         min_abs_diag = std::min(min_abs_diag, diag_abs);
         const double denom = (off_abs_sum > 0.0) ? off_abs_sum : 1e-30;
         diag_dom_sum += diag_abs / denom;
         diag_dom_count++;
      }
   }

   out[F_MAX_NNZ] = static_cast<double>(max_row_nnz);
   out[F_FROB_NORM] = std::sqrt(frob_sq);
   out[F_DIAG_DOMINANCE] =
      (diag_dom_count > 0) ? diag_dom_sum / diag_dom_count : 0.0;
   out[F_COND_EST] = (min_abs_diag > 0.0 && std::isfinite(min_abs_diag))
                        ? max_abs_diag / min_abs_diag
                        : 0.0;

   // Symmetry: scan upper triangle and look up A(j,i). Use fabs differences,
   // accept O(nnz log row) cost since the largest matrices we collect are not
   // huge.
   double asym_sq = 0.0;
   for (int i = 0; i < n; ++i)
   {
      for (int k = I[i]; k < I[i + 1]; ++k)
      {
         const int j = J[k];
         if (j <= i) { continue; }
         const double a_ij = V[k];
         double a_ji = 0.0;
         for (int kk = I[j]; kk < I[j + 1]; ++kk)
         {
            if (J[kk] == i) { a_ji = V[kk]; break; }
         }
         const double d = a_ij - a_ji;
         asym_sq += 2.0 * d * d; // both halves of the off-diagonal pair
      }
   }
   out[F_SYMMETRY] =
      (out[F_FROB_NORM] > 0.0) ? std::sqrt(asym_sq) / out[F_FROB_NORM] : 0.0;
}

inline void WriteCSVHeader(std::ostream &os, int num_solvers)
{
   for (int i = 0; i < F_COUNT; ++i)
   {
      os << kFeatureNames[i] << ',';
   }
   for (int s = 0; s < num_solvers; ++s)
   {
      os << "t_solver_" << s;
      if (s + 1 < num_solvers) { os << ','; }
   }
   os << '\n';
}

inline void WriteCSVRow(std::ostream &os, const FeatureVector &f,
                        const std::vector<double> &times)
{
   for (int i = 0; i < F_COUNT; ++i)
   {
      os << f[i] << ',';
   }
   for (size_t s = 0; s < times.size(); ++s)
   {
      os << times[s];
      if (s + 1 < times.size()) { os << ','; }
   }
   os << '\n';
}

} // namespace ai_ss

#endif // AI_SS_MATRIX_FEATURES_HPP
