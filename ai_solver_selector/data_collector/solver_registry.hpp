// AI Solver Selector - registry of (Krylov solver, preconditioner) pairs.
//
// The order of entries here defines the integer label that the trained model
// predicts. Keep it stable; append new entries at the end.

#ifndef AI_SS_SOLVER_REGISTRY_HPP
#define AI_SS_SOLVER_REGISTRY_HPP

#include "mfem.hpp"
#include <memory>
#include <string>
#include <vector>

namespace ai_ss
{

enum SolverId
{
   SOLVER_CG_JACOBI = 0,
   SOLVER_CG_GS,
   SOLVER_GMRES_JACOBI,
   SOLVER_GMRES_GS,
   SOLVER_BICGSTAB_JACOBI,
   SOLVER_MINRES_JACOBI,
   SOLVER_COUNT
};

inline const char *SolverName(int id)
{
   static const char *kNames[SOLVER_COUNT] = {
      "CG+Jacobi",     "CG+GS",         "GMRES+Jacobi",
      "GMRES+GS",      "BiCGStab+Jacobi", "MINRES+Jacobi"
   };
   return (id >= 0 && id < SOLVER_COUNT) ? kNames[id] : "unknown";
}

// Configure a solver and its preconditioner for the given matrix. The caller
// owns both pointers.
struct SolverPair
{
   std::unique_ptr<mfem::IterativeSolver> solver;
   std::unique_ptr<mfem::Solver> prec;
};

inline SolverPair MakeSolver(int id, mfem::SparseMatrix &A,
                             int max_iter, double rtol)
{
   SolverPair sp;
   switch (id)
   {
      case SOLVER_CG_JACOBI:
         sp.prec.reset(new mfem::DSmoother(A, 0, 1.0, 1));
         sp.solver.reset(new mfem::CGSolver());
         break;
      case SOLVER_CG_GS:
         sp.prec.reset(new mfem::GSSmoother(A));
         sp.solver.reset(new mfem::CGSolver());
         break;
      case SOLVER_GMRES_JACOBI:
         sp.prec.reset(new mfem::DSmoother(A, 0, 1.0, 1));
         sp.solver.reset(new mfem::GMRESSolver());
         break;
      case SOLVER_GMRES_GS:
         sp.prec.reset(new mfem::GSSmoother(A));
         sp.solver.reset(new mfem::GMRESSolver());
         break;
      case SOLVER_BICGSTAB_JACOBI:
         sp.prec.reset(new mfem::DSmoother(A, 0, 1.0, 1));
         sp.solver.reset(new mfem::BiCGSTABSolver());
         break;
      case SOLVER_MINRES_JACOBI:
         sp.prec.reset(new mfem::DSmoother(A, 0, 1.0, 1));
         sp.solver.reset(new mfem::MINRESSolver());
         break;
      default:
         MFEM_ABORT("Unknown solver id");
   }

   sp.solver->SetRelTol(rtol);
   sp.solver->SetAbsTol(0.0);
   sp.solver->SetMaxIter(max_iter);
   sp.solver->SetPrintLevel(-1);
   sp.solver->SetPreconditioner(*sp.prec);
   sp.solver->SetOperator(A);
   return sp;
}

} // namespace ai_ss

#endif // AI_SS_SOLVER_REGISTRY_HPP
