import pytest
import os
from solver_engine.maze_utils import Mazes
from solver_engine.results import MazeSolution
from solver_engine.traditional_solvers.collision_solver import MazelibCollisionSolver
from solver_engine.traditional_solvers.backtracking_solver import MazelibBacktrackingSolver

# Get the absolute path to the project root (assuming tests are run from root or one level down)
# This makes the test robust to where pytest is called from.
# However, for simplicity with current setup, we might just use relative path "mazes/maze_5x5_1.json"
# if workspace root is consistently where pytest is run.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

@pytest.fixture
def maze_5x5_from_file():
    # Load a real 5x5 maze file.
    # Assuming 'mazes' directory is at the project root.
    maze_file_path = os.path.join(PROJECT_ROOT, "mazes", "maze_5x5_1.json")
    # If pytest is always run from workspace root, this could be simpler:
    # maze_file_path = "mazes/maze_5x5_1.json"
    if not os.path.exists(maze_file_path):
        pytest.skip(f"Maze file not found: {maze_file_path}. Ensure 'mazes' dir is populated and at project root.")
    return Mazes.load(maze_file_path)

def test_mazelib_collision_solver_on_file_maze(maze_5x5_from_file):
    maze = maze_5x5_from_file
    solver = MazelibCollisionSolver()

    expected_start = (1, 0)
    expected_end = (maze.grid.shape[0] - 2, maze.grid.shape[1] - 1)

    solution = solver.solve(maze)

    assert isinstance(solution, MazeSolution)
    assert solution.solver_name == "MazelibCollisionSolver"
    assert solution.maze_id == (maze.size, maze.maze_number) 
    # This assertion might still fail if Collision solver doesn't solve maze_5x5_1.json
    assert solution.valid is True, f"CollisionSolver failed for {solution.maze_id}, path: {solution.path}, metadata: {solution.metadata}"
    assert solution.solve_time >= 0

    if solution.valid:
        assert solution.path is not None
        assert solution.path[0] == expected_start, "Path does not start at the expected point"
        assert solution.path[-1] == expected_end, "Path does not end at the expected point"

def test_mazelib_backtracking_solver_on_file_maze(maze_5x5_from_file):
    maze = maze_5x5_from_file
    solver = MazelibBacktrackingSolver()

    expected_start = (1, 0)
    expected_end = (maze.grid.shape[0] - 2, maze.grid.shape[1] - 1)

    solution = solver.solve(maze)

    assert isinstance(solution, MazeSolution)
    assert solution.solver_name == "MazelibBacktrackingSolver"
    assert solution.maze_id == (maze.size, maze.maze_number)
    assert solution.valid is True, f"BacktrackingSolver failed for {solution.maze_id}, path: {solution.path}, metadata: {solution.metadata}"
    assert solution.solve_time >= 0

    if solution.valid:
        assert solution.path is not None
        assert solution.path[0] == expected_start, "Path does not start at the expected point"
        assert solution.path[-1] == expected_end, "Path does not end at the expected point"
