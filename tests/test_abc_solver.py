import pytest
import numpy as np
from solver_engine.abc_solver import Solver
from solver_engine.maze_utils import Mazes
from solver_engine.results import MazeSolution

# A minimal concrete implementation for testing the ABC
class DummySolver(Solver):
    def solve(self, maze: Mazes, **kwargs) -> MazeSolution:
        # Dummy implementation for testing purposes
        return MazeSolution(
            solver_name="DummySolver",
            maze_id=(maze.size, maze.maze_number),
            path=[(1,0), (1,1), (2,1)], # Dummy path
            valid=True,
            solve_time=0.01,
            metadata={"info": "dummy solve"}
        )

@pytest.fixture
def dummy_maze():
    grid = np.array([[1,1,1],[0,0,0],[1,1,1]])
    return Mazes(size=3, maze_number=1, grid=grid, generation_time=0.0)

def test_solver_interface(dummy_maze):
    solver = DummySolver()
    solution = solver.solve(dummy_maze)

    assert isinstance(solution, MazeSolution)
    assert solution.solver_name == "DummySolver"
    assert solution.maze_id == (3,1)
    assert isinstance(solution.path, list)
    assert solution.valid is True
    assert solution.solve_time >= 0
    assert "info" in solution.metadata 