import pytest
import os
import json
from unittest.mock import MagicMock, call # Added call for checking call_args
import numpy as np

from solver_engine.experiment_runner import ExperimentRunner
from solver_engine.abc_solver import Solver # For DummySolver type hint
from solver_engine.maze_utils import Mazes
from solver_engine.results import MazeSolution
from solver_engine.llm_solvers.input_formatters import AsciiInputFormatter # For LLM Dummy

# --- Test Fixtures & Dummies ---

@pytest.fixture
def temp_maze_files(tmp_path):
    maze_files = []
    maze_data_1 = {"size": 3, "maze_number": 1, "grid": [[1,1,1],[0,0,0],[1,1,1]]}
    maze_data_2 = {"size": 3, "maze_number": 2, "grid": [[1,0,1],[0,0,0],[1,0,1]]}
    
    file1_path = tmp_path / "maze1.json"
    with open(file1_path, 'w') as f1:
        json.dump(maze_data_1, f1)
    maze_files.append(str(file1_path))
    
    file2_path = tmp_path / "maze2.json"
    with open(file2_path, 'w') as f2:
        json.dump(maze_data_2, f2)
    maze_files.append(str(file2_path))
    
    return maze_files

class DummySolverImpl(Solver):
    def __init__(self, solver_id: str, should_succeed: bool = True, solve_time_val: float = 0.1):
        self._solver_name = solver_id
        self.should_succeed = should_succeed
        self.solve_time_val = solve_time_val
        self.solve_call_count = 0
        self.solve_call_args = [] # Store args for verification

    @property
    def solver_name(self) -> str:
        return self._solver_name

    def solve(self, maze: Mazes, **kwargs) -> MazeSolution:
        self.solve_call_count += 1
        self.solve_call_args.append({'maze': maze, 'kwargs': kwargs})
        return MazeSolution(
            solver_name=self.solver_name,
            maze_id=(maze.size, maze.maze_number),
            path=[(1,0)] if self.should_succeed else [],
            valid=self.should_succeed,
            solve_time=self.solve_time_val,
            metadata={'dummy_data': 'solved' if self.should_succeed else 'failed'}
        )

@pytest.fixture
def dummy_solvers():
    return [
        DummySolverImpl(solver_id="DummySolverA", should_succeed=True),
        DummySolverImpl(solver_id="DummySolverB", should_succeed=False, solve_time_val=0.05)
    ]

# --- Tests for ExperimentRunner ---

def test_experiment_runner_initialization(dummy_solvers, temp_maze_files):
    runner = ExperimentRunner(solvers=dummy_solvers, maze_files=temp_maze_files)
    assert runner.solvers == dummy_solvers
    assert runner.maze_files == temp_maze_files
    assert runner.maze_loader == Mazes.load

def test_experiment_runner_init_no_solvers(temp_maze_files):
    with pytest.raises(ValueError, match="At least one solver must be provided."):
        ExperimentRunner(solvers=[], maze_files=temp_maze_files)

def test_experiment_runner_init_no_maze_files(dummy_solvers):
    with pytest.raises(ValueError, match="At least one maze file must be provided."):
        ExperimentRunner(solvers=dummy_solvers, maze_files=[])

def test_run_experiments_successful_run(dummy_solvers, temp_maze_files):
    runner = ExperimentRunner(solvers=dummy_solvers, maze_files=temp_maze_files)
    results = runner.run_experiments()

    assert len(results) == len(temp_maze_files) * len(dummy_solvers) # 2 mazes * 2 solvers = 4 results

    # Check calls to solvers
    assert dummy_solvers[0].solve_call_count == len(temp_maze_files) # SolverA called for each maze
    assert dummy_solvers[1].solve_call_count == len(temp_maze_files) # SolverB called for each maze

    # Check some result properties
    for result in results:
        assert isinstance(result, MazeSolution)
        if result.solver_name == "DummySolverA":
            assert result.valid is True
            assert result.solve_time == 0.1
        elif result.solver_name == "DummySolverB":
            assert result.valid is False
            assert result.solve_time == 0.05
    
    # Check that solve was called with the correct Mazes objects
    # For SolverA
    loaded_maze1 = Mazes.load(temp_maze_files[0])
    loaded_maze2 = Mazes.load(temp_maze_files[1])
    
    solver_a_mazes_solved = [args['maze'].maze_number for args in dummy_solvers[0].solve_call_args]
    assert loaded_maze1.maze_number in solver_a_mazes_solved
    assert loaded_maze2.maze_number in solver_a_mazes_solved

def test_run_experiments_maze_load_error(dummy_solvers, temp_maze_files, tmp_path, capsys):
    bad_maze_file = tmp_path / "bad_maze.json"
    bad_maze_file.write_text("this is not json")
    all_files = [str(bad_maze_file)] + temp_maze_files

    runner = ExperimentRunner(solvers=dummy_solvers, maze_files=all_files)
    results = runner.run_experiments()

    # Only valid mazes should result in solver calls (2 valid mazes * 2 solvers = 4 results)
    # The bad maze load is skipped by the runner, no error MazeSolutions are created for load errors by default
    assert len(results) == len(temp_maze_files) * len(dummy_solvers)
    assert dummy_solvers[0].solve_call_count == len(temp_maze_files)
    captured = capsys.readouterr()
    assert f"Error loading maze {str(bad_maze_file)}" in captured.out

def test_run_experiments_solver_execution_error(temp_maze_files):
    # Create a solver that will raise an exception
    error_solver = MagicMock(spec=Solver)
    error_solver.solver_name = "ErrorSolver"
    error_solver.solve.side_effect = Exception("Solver crashed!")
    
    normal_solver = DummySolverImpl(solver_id="NormalSolver")
    
    solvers_with_error = [error_solver, normal_solver]
    maze_file_list = [temp_maze_files[0]] # Use only one maze for simplicity

    runner = ExperimentRunner(solvers=solvers_with_error, maze_files=maze_file_list)
    results = runner.run_experiments()

    assert len(results) == 2 # One error solution, one normal solution
    error_result_found = False
    normal_result_found = False

    for res in results:
        if res.solver_name == "ErrorSolver":
            error_result_found = True
            assert res.valid is False
            assert res.path == []
            assert res.metadata['error'] == "Solver crashed!"
        elif res.solver_name == "NormalSolver":
            normal_result_found = True
            assert res.valid is True # DummySolverImpl defaults to True

    assert error_result_found
    assert normal_result_found
    error_solver.solve.assert_called_once() # Ensure it was called
    # normal_solver.solve_call_count should be 1, but normal_solver isn't the MagicMock 