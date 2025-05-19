import time
import numpy as np
from abc import abstractmethod
from mazelib import Maze as MazelibMaze
from typing import List, Tuple, Any

from solver_engine.abc_solver import Solver
from solver_engine.maze_utils import Mazes
from solver_engine.results import MazeSolution

class MazelibSolverBase(Solver):
    """Base class for mazelib-based maze solvers."""

    @property
    @abstractmethod
    def solver_name(self) -> str:
        """The specific name of the mazelib solver."""
        pass

    @abstractmethod
    def _get_mazelib_solver_instance(self) -> Any:
        """Return an instance of the specific mazelib solver."""
        pass

    def solve(self, maze: Mazes, **kwargs) -> MazeSolution:
        """Solves the maze using a configured mazelib solver."""
        start_time = time.time()

        m = MazelibMaze()
        m.grid = maze.grid.copy() # Mazelib might modify the grid

        # Standard start/end points for our mazes
        defined_start_point = (1, 0)
        defined_end_point = (maze.grid.shape[0] - 2, maze.grid.shape[1] - 1)

        m.start = defined_start_point
        m.end = defined_end_point

        mazelib_solver_instance = self._get_mazelib_solver_instance()
        m.solver = mazelib_solver_instance
        
        m.solve()

        solve_time = time.time() - start_time

        solution_path: List[Tuple[int, int]] = []
        is_valid = False

        # pre-pend and append the start and end points to each path in m.solutions
        for path in m.solutions:
            path.insert(0, defined_start_point)
            path.append(defined_end_point)

        # we only need one solution
        if m.solutions and m.solutions[0]:
            solution_path = [tuple(point) for point in m.solutions[0]]
            is_valid = True
        else:
            is_valid = False

        return MazeSolution(
            solver_name=self.solver_name,
            maze_id=(maze.size, maze.maze_number),
            path=solution_path,
            valid=is_valid,
            solve_time=solve_time,
            metadata={'mazelib_solutions_count': len(m.solutions) if m.solutions else 0}
        ) 