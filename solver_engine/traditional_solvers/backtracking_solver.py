from mazelib.solve.BacktrackingSolver import BacktrackingSolver
from solver_engine.traditional_solvers.base import MazelibSolverBase
from typing import Any

class MazelibBacktrackingSolver(MazelibSolverBase):
    """Wrapper for the mazelib.solve.BacktrackingSolver.BacktrackingSolver."""

    @property
    def solver_name(self) -> str:
        return "MazelibBacktrackingSolver"

    def _get_mazelib_solver_instance(self) -> Any:
        return BacktrackingSolver() 