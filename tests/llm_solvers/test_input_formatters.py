import pytest
import numpy as np
from solver_engine.llm_solvers.input_formatters import AsciiInputFormatter
from solver_engine.llm_solvers.input_formatters import EmojiInputFormatter

@pytest.fixture
def sample_grid_3x3():
    return np.array([
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

def test_ascii_input_formatter(sample_grid_3x3):
    formatter = AsciiInputFormatter()
    assert formatter.wall_char == "#"
    assert formatter.path_char == "."

    expected_map_string = (
        "##.\n"
        ".#.\n"
        "..#"
    )
    assert formatter.format_grid(sample_grid_3x3) == expected_map_string

def test_emoji_input_formatter(sample_grid_3x3):
    formatter = EmojiInputFormatter()
    assert formatter.wall_char == "🧱"
    assert formatter.path_char == "➡️"

    expected_map_string = (
        "🧱🧱➡️\n"
        "➡️🧱➡️\n"
        "➡️➡️🧱"
    )
    assert formatter.format_grid(sample_grid_3x3) == expected_map_string 