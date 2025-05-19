import pytest
import os
from unittest.mock import patch, MagicMock

from solver_engine.llm_solvers.openai_solver import OpenAISolver
from solver_engine.llm_solvers.input_formatters import AsciiInputFormatter
from solver_engine.maze_utils import Mazes # For creating a dummy maze object
import numpy as np # For dummy maze grid

# Dummy API Key for testing - ensure this is set in your test environment if needed,
# or tests might skip if OPENAI_API_KEY env var is strictly required by constructor.
# For these tests, we primarily mock, so the actual key value for client init might not matter
# as long as it passes the constructor's check if it expects a non-empty string.

@pytest.fixture
def ascii_formatter_openai():
    return AsciiInputFormatter()

@pytest.fixture
def dummy_maze_for_openai():
    # A simple 3x3 maze for basic testing of the solve loop
    grid = np.array([[1,1,1],[0,0,0],[1,1,1]]) # Start (1,0), End (1,2)
    return Mazes(size=3, maze_number=1, grid=grid)

@pytest.fixture(autouse=True)
def ensure_dummy_openai_api_key(monkeypatch):
    """Ensure OPENAI_API_KEY is set to a dummy value for tests if not already set."""
    if not os.getenv("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-dummykeyfortest")

def test_openai_solver_initialization(ascii_formatter_openai):
    solver = OpenAISolver(formatter=ascii_formatter_openai, model_name="gpt-test")
    assert solver.model_name == "gpt-test"
    assert solver.solver_name == "OpenAISolver-gpt-test"
    assert solver.client is not None

def test_openai_solver_initialization_no_api_key(monkeypatch, ascii_formatter_openai):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OpenAI API key not provided"):
        OpenAISolver(formatter=ascii_formatter_openai, api_key=None) # Explicitly pass None

@patch('openai.resources.chat.completions.Completions.create')
def test_openai_solver_make_api_call_success(mock_create, ascii_formatter_openai):
    solver = OpenAISolver(formatter=ascii_formatter_openai, model_name="gpt-test")
    
    # Mock OpenAI API response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = " [(1,0), (1,1)] "
    mock_create.return_value = mock_response
    
    response_text = solver._make_api_call("Test prompt")
    
    assert response_text == "[(1,0), (1,1)]"
    mock_create.assert_called_once_with(
        model="gpt-test",
        messages=[
            {"role": "system", "content": "You are an expert maze solver. Follow the output rules precisely."},
            {"role": "user", "content": "Test prompt"}
        ],
    )

@patch('openai.resources.chat.completions.Completions.create')
def test_openai_solver_make_api_call_failure(mock_create, ascii_formatter_openai):
    solver = OpenAISolver(formatter=ascii_formatter_openai, model_name="gpt-test")
    mock_create.side_effect = Exception("API Error")
    
    response_text = solver._make_api_call("Test prompt")
    
    assert response_text == "" # Should return empty string on error

@patch('openai.resources.chat.completions.Completions.create')
def test_openai_solver_solve_method_mocked_success(mock_create, ascii_formatter_openai, dummy_maze_for_openai):
    solver = OpenAISolver(formatter=ascii_formatter_openai, model_name="gpt-test", max_trials=1)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    # Path for 3x3 maze: (1,0) -> (1,1) -> (1,2)
    mock_response.choices[0].message.content = "[(1,0), (1,1), (1,2)]"
    mock_create.return_value = mock_response

    solution = solver.solve(dummy_maze_for_openai)

    assert solution.valid is True
    assert solution.path == [(1,0), (1,1), (1,2)]
    assert mock_create.call_count == 1 # Called once
    assert len(solution.metadata['llm_history']) == 1
    assert solution.metadata['llm_raw_response'] == "[(1,0), (1,1), (1,2)]" 