from abc import ABC, abstractmethod
from solver_engine.maze_utils import Mazes
from solver_engine.results import MazeSolution
from typing import Any, Dict

class Solver(ABC):
    """Abstract Base Class for all maze solvers."""

    @abstractmethod
    def solve(self, maze: Mazes, **kwargs: Any) -> MazeSolution:
        """
        Solves the given maze.

        Args:
            maze: The Mazes object to be solved.
            **kwargs: Additional solver-specific arguments.

        Returns:
            A MazeSolution object containing the solution details.
        """
        pass 