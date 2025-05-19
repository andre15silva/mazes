import pytest
import os
import numpy as np
from solver_engine.maze_utils import Mazes # This will fail initially

@pytest.fixture
def sample_maze_file(tmp_path):
    content = '''{
      "size": 5,
      "maze_number": 1,
      "grid": [
        [1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1]
      ],
      "generation_time": 0.01
    }'''
    file_path = tmp_path / "test_maze.json"
    file_path.write_text(content)
    return str(file_path)

def test_load_valid_maze(sample_maze_file):
    maze = Mazes.load(sample_maze_file)
    assert maze is not None
    assert maze.size == 5
    assert maze.maze_number == 1
    assert isinstance(maze.grid, np.ndarray)
    assert maze.grid.shape == (5, 5)
    expected_grid = np.array([
        [1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1]
    ])
    np.testing.assert_array_equal(maze.grid, expected_grid)
    assert maze.generation_time == 0.01


@pytest.fixture
def sample_maze_instance():
    grid = np.array([
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1]
    ], dtype=int)
    return Mazes(size=5, maze_number=2, grid=grid, generation_time=0.05)

def test_save_and_load_maze(sample_maze_instance, tmp_path):
    maze_original = sample_maze_instance
    save_path = tmp_path / "saved_maze.json"

    maze_original.save(str(save_path))

    assert save_path.exists(), "Maze file was not saved."

    maze_loaded = Mazes.load(str(save_path))

    assert maze_loaded is not None
    assert maze_loaded.size == maze_original.size
    assert maze_loaded.maze_number == maze_original.maze_number
    np.testing.assert_array_equal(maze_loaded.grid, maze_original.grid)
    assert maze_loaded.generation_time == pytest.approx(maze_original.generation_time) 