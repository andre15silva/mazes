from abc import ABC, abstractmethod
import numpy as np

class InputFormatter(ABC):
    """Abstract Base Class for formatting maze grids for LLM prompts."""

    @property
    @abstractmethod
    def wall_char(self) -> str:
        """Character representation for a wall."""
        pass

    @property
    @abstractmethod
    def path_char(self) -> str:
        """Character representation for a passable cell."""
        pass

    @abstractmethod
    def format_grid(self, grid: np.ndarray) -> str:
        """
        Formats the maze grid into a string representation using wall_char and path_char.

        Args:
            grid: A NumPy array representing the maze (1 for wall, 0 for path).

        Returns:
            A string representation of the maze map.
        """
        pass

class AsciiInputFormatter(InputFormatter):
    """Formats the maze using '#' for walls and '.' for paths."""

    @property
    def wall_char(self) -> str:
        return "#"

    @property
    def path_char(self) -> str:
        return "."

    def format_grid(self, grid: np.ndarray) -> str:
        rows = [''.join(self.wall_char if cell == 1 else self.path_char for cell in row) for row in grid]
        return "\n".join(rows)

class EmojiInputFormatter(InputFormatter):
    """Formats the maze using '🧱' for walls and '➡️' for paths."""

    @property
    def wall_char(self) -> str:
        return "🧱" # Brick emoji for wall

    @property
    def path_char(self) -> str:
        return "➡️" # Right arrow for path (could be any path-like emoji)

    def format_grid(self, grid: np.ndarray) -> str:
        rows = [''.join(self.wall_char if cell == 1 else self.path_char for cell in row) for row in grid]
        return "\n".join(rows) 