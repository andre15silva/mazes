import pytest
import numpy as np
import os # Added for path joining
from typing import List, Dict, Any, Tuple

from solver_engine.maze_utils import Mazes
from solver_engine.results import MazeSolution
from solver_engine.llm_solvers.input_formatters import AsciiInputFormatter
from solver_engine.llm_solvers.abc_llm_solver import LLMSolverBase

# --- Helper to load maze for tests ---
PROJECT_ROOT_LLM_TEST = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MAZE_FILE_5X5 = os.path.join(PROJECT_ROOT_LLM_TEST, "mazes", "maze_5x5_1.json")

def load_test_maze(file_path=MAZE_FILE_5X5) -> Mazes:
    if not os.path.exists(file_path):
        pytest.skip(f"Maze file not found: {file_path}. Ensure 'mazes' dir is populated.")
    return Mazes.load(file_path)

# --- Test Fixtures ---
# Removed solvable_maze_3x3_llm fixture

@pytest.fixture
def ascii_formatter_llm():
    return AsciiInputFormatter()

# --- Dummy LLM Solver for Testing --- 
class MockLLMProviderSolver(LLMSolverBase):
    def __init__(self, formatter: AsciiInputFormatter, max_trials: int = 3, responses: List[str] = None):
        super().__init__(formatter, max_trials)
        self.api_call_count = 0
        self.responses = responses if responses else []
        self._solver_name_prop = "MockLLMProviderSolver"

    @property
    def solver_name(self) -> str:
        return self._solver_name_prop
    
    def set_solver_name(self, name: str):
        self._solver_name_prop = name

    def _make_api_call(self, prompt: str) -> str:
        if self.api_call_count < len(self.responses):
            response = self.responses[self.api_call_count]
            self.api_call_count += 1
            return response
        return "[]" # Default empty response if out of predefined responses

# --- Tests --- 

def test_llm_solver_base_initialization(ascii_formatter_llm):
    solver = MockLLMProviderSolver(formatter=ascii_formatter_llm, max_trials=5)
    assert solver.formatter == ascii_formatter_llm
    assert solver.max_trials == 5
    assert solver.solver_name == "MockLLMProviderSolver"

def test_llm_solver_base_parse_response():
    solver = MockLLMProviderSolver(formatter=AsciiInputFormatter()) 
    assert solver._parse_response("[(1,0), (1,1)]") == [(1,0), (1,1)]
    assert solver._parse_response("  [(1, 0), (1, 1)]  ") == [(1,0), (1,1)]
    assert solver._parse_response("Blah blah [(1,0),(2,0)] blah") == [(1,0),(2,0)]
    assert solver._parse_response("[(1,0), (1,1),(1,2)]") == [(1,0), (1,1), (1,2)]
    assert solver._parse_response("Path: [(0,1), (1,1), (2,1)]") == [(0,1), (1,1), (2,1)]
    assert solver._parse_response("No path found. []") == []
    assert solver._parse_response("[]") == []
    assert solver._parse_response("[(0,0), (0,1), (0,2), (1,2), (2,2), (2,1), (2,0)]") == [(0,0), (0,1), (0,2), (1,2), (2,2), (2,1), (2,0)]
    assert solver._parse_response("Sure, the path is: [(1,0), (1,1), (2,1)]") == [(1,0), (1,1), (2,1)]
    assert solver._parse_response("[(1, 0), (1, 1), (1, 2)]") == [(1,0), (1,1), (1,2)]
    assert solver._parse_response("[(1,0)]") == [(1,0)]
    assert solver._parse_response("[(0,0),(1,0)]\nSome other text") == [(0,0),(1,0)]
    assert solver._parse_response("Absolutely nothing useful") == []
    assert solver._parse_response("Here is one: (1,0),(2,0),(3,0) but it is not a list.") == [(1,0),(2,0),(3,0)]

def test_llm_solver_successful_first_try(ascii_formatter_llm):
    test_maze = load_test_maze() # Load 5x5 maze
    print(test_maze.grid)
    # For a 5x5 maze, start=(1,0), end=(3,4). A possible path for maze_5x5_1.json
    # This is an example path, the actual path from maze_5x5_1.json might differ
    # and a mock LLM solver doesn't actually solve it. We just need a valid formatted path.
    # Let's use a short, simple, but valid path structure for the mock response.
    mock_valid_path_5x5 = [(1,0), (1,1), (2,1), (3,1), (3, 2), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (8, 3),
                           (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10)] 
    responses = [str(mock_valid_path_5x5)]
    solver = MockLLMProviderSolver(formatter=ascii_formatter_llm, responses=responses, max_trials=1)
    
    solution = solver.solve(test_maze)

    assert solution.valid is True
    assert solution.path == mock_valid_path_5x5
    assert solution.solve_time >= 0
    assert solver.api_call_count == 1
    assert len(solution.metadata['llm_history']) == 1
    assert solution.metadata['trials_taken'] == 1

def test_llm_solver_fail_then_succeed(ascii_formatter_llm):
    test_maze = load_test_maze() # Load 5x5 maze
    # For a 5x5 maze, start=(1,0), end=(3,4)
    mock_valid_path_5x5 = [(1,0), (1,1), (2,1), (3,1), (3, 2), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (8, 3),
                           (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10)] 
    responses = [
        "[(1, 0), (0, 0), (0, 1)]", # Invalid move from (1,0) to (0,0) (hits outer wall or invalid move)
        str(mock_valid_path_5x5)  # Valid path
    ]
    solver = MockLLMProviderSolver(formatter=ascii_formatter_llm, responses=responses, max_trials=2)

    solution = solver.solve(test_maze)

    assert solution.valid is True
    assert solution.path == mock_valid_path_5x5
    assert solver.api_call_count == 2
    assert len(solution.metadata['llm_history']) == 2
    assert solution.metadata['trials_taken'] == 2
    history_entry_2 = solution.metadata['llm_history'][1]
    assert "Here is a previous attempt (Attempt #1)" in history_entry_2['prompt']
    assert "[(1, 0), (0, 0)]" in history_entry_2['prompt']

def test_llm_solver_max_trials_reached(ascii_formatter_llm):
    test_maze = load_test_maze()
    responses = [
        "[(0,0)]", # Invalid start
        "[(1,0), (5,0)]", # Out of bounds for 5x5
        "[(1,0), (1,0), (1,1)]"  # Revisits cell (1,0)
    ]
    solver = MockLLMProviderSolver(formatter=ascii_formatter_llm, responses=responses, max_trials=3)

    solution = solver.solve(test_maze)

    assert solution.valid is False
    assert solution.path == [] 
    assert solver.api_call_count == 3
    assert len(solution.metadata['llm_history']) == 3
    assert solution.metadata['trials_taken'] == 3

def test_llm_solver_prompt_formatting(ascii_formatter_llm):
    test_maze = load_test_maze()
    solver = MockLLMProviderSolver(formatter=ascii_formatter_llm, responses=["[]"])
    prompt1 = solver._format_prompt(test_maze, [])
    print(prompt1)
    assert "You are given a square maze" in prompt1
    assert "Start position: (1, 0)" in prompt1
    assert f"End position:   ({test_maze.size*2-1}, {test_maze.size*2})" in prompt1
    assert f"  - Walls: {ascii_formatter_llm.wall_char} (impassable)" in prompt1
    assert f"  - Free cells: {ascii_formatter_llm.path_char} (traversable)" in prompt1
    assert ascii_formatter_llm.format_grid(test_maze.grid) in prompt1
    assert "Here is a previous attempt" not in prompt1

    history_for_prompt = [
        {
            'path': [(1,0), (0,0)], 
            'invalid_first': [1], 
            'invalid_all': [1]
        }
    ]
    prompt2 = solver._format_prompt(test_maze, history_for_prompt)
    assert "Here is a previous attempt (Attempt #1)" in prompt2
    assert "[(1, 0), (0, 0)]" in prompt2 
    assert "The last point shown was where the path became invalid" in prompt2 