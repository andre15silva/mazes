import pytest
import os
import numpy as np
from solver_engine.maze_utils import Mazes
from solver_engine.results import MazeSolution
from solver_engine.plotting_utils import plot_solution_to_file

@pytest.fixture
def dummy_plot_maze():
    grid = np.array([[1,1,1],[0,0,0],[1,0,1]])
    return Mazes(size=3, maze_number=1, grid=grid)

@pytest.fixture
def dummy_plot_solution_valid(dummy_plot_maze):
    return MazeSolution(
        solver_name="PlotTestSolverValid",
        maze_id=(dummy_plot_maze.size, dummy_plot_maze.maze_number),
        path=[(1,0),(1,1),(1,2)], # Valid path for 3x3 with (1,0) start, (1,2) end
        valid=True,
        solve_time=0.1,
        metadata={}
    )

@pytest.fixture
def dummy_plot_solution_invalid_path(dummy_plot_maze):
    return MazeSolution(
        solver_name="PlotTestSolverInvalidPath",
        maze_id=(dummy_plot_maze.size, dummy_plot_maze.maze_number),
        path=[(1,0),(0,0),(1,2)], # Invalid step (0,0) is wall
        valid=False,
        solve_time=0.1,
        metadata={}
    )

@pytest.fixture
def dummy_plot_solution_no_path(dummy_plot_maze):
    return MazeSolution(
        solver_name="PlotTestSolverNoPath",
        maze_id=(dummy_plot_maze.size, dummy_plot_maze.maze_number),
        path=[], 
        valid=False,
        solve_time=0.1,
        metadata={}
    )

def test_plot_solution_to_file_runs_valid(dummy_plot_maze, dummy_plot_solution_valid, tmp_path):
    output_directory = tmp_path / "plots"
    plot_file = plot_solution_to_file(dummy_plot_maze, dummy_plot_solution_valid, str(output_directory))
    assert plot_file is not None
    assert os.path.exists(plot_file)
    assert "PlotTestSolverValid" in plot_file
    assert f"maze_{dummy_plot_maze.size}x{dummy_plot_maze.size}_{dummy_plot_maze.maze_number}" in plot_file
    assert "_final.pdf" in plot_file

def test_plot_solution_to_file_runs_invalid_path(dummy_plot_maze, dummy_plot_solution_invalid_path, tmp_path):
    output_directory = tmp_path / "plots"
    # Provide invalid_indices based on the known bad path for more specific coloring
    invalid_indices_for_plot = [1] # (0,0) is the 2nd point (index 1) and is bad
    plot_file = plot_solution_to_file(
        dummy_plot_maze, 
        dummy_plot_solution_invalid_path, 
        str(output_directory),
        invalid_indices=invalid_indices_for_plot
    )
    assert plot_file is not None
    assert os.path.exists(plot_file)
    assert "PlotTestSolverInvalidPath" in plot_file

def test_plot_solution_to_file_runs_no_path(dummy_plot_maze, dummy_plot_solution_no_path, tmp_path):
    output_directory = tmp_path / "plots"
    plot_file = plot_solution_to_file(dummy_plot_maze, dummy_plot_solution_no_path, str(output_directory))
    assert plot_file is not None
    assert os.path.exists(plot_file)
    assert "PlotTestSolverNoPath" in plot_file

def test_plot_solution_with_attempt_number(dummy_plot_maze, dummy_plot_solution_valid, tmp_path):
    output_directory = tmp_path / "plots"
    attempt_num = 3
    plot_file = plot_solution_to_file(dummy_plot_maze, dummy_plot_solution_valid, str(output_directory), attempt_number=attempt_num)
    assert plot_file is not None
    assert os.path.exists(plot_file)
    assert f"_attempt_{attempt_num}.pdf" in plot_file

# It might be good to test specific coloring logic if we can inspect the plot object itself,
# but that is more involved. For now, ensuring it runs and creates files is a good start. 