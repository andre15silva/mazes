from mazelib.solve.Collision import Collision
from solver_engine.traditional_solvers.base import MazelibSolverBase
from typing import Any

class MazelibCollisionSolver(MazelibSolverBase):
    """Wrapper for the mazelib.solve.Collision.Collision solver."""

    @property
    def solver_name(self) -> str:
        return "MazelibCollisionSolver"

    def _get_mazelib_solver_instance(self) -> Any:
        return Collision() 