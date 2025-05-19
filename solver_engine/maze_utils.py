import json
import numpy as np
from typing import List, Any # Added List for type hint

class Mazes:
    def __init__(self, size: int, maze_number: int, grid: np.ndarray, generation_time: float = 0.0):
        self.size = size
        self.maze_number = maze_number
        self.grid = grid
        self.generation_time = generation_time

    @classmethod
    def load(cls, filename: str) -> 'Mazes':
        """Load a maze from a JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        grid_data = data.get("grid")
        if not isinstance(grid_data, list):
            raise ValueError("Maze grid in JSON is not a list.")
        
        # Ensure all rows have the same length and all elements are numbers
        if grid_data:
            first_row_len = len(grid_data[0])
            for row in grid_data:
                if not isinstance(row, list) or len(row) != first_row_len:
                    raise ValueError("Maze grid rows are not lists or have inconsistent lengths.")
                for cell in row:
                    if not isinstance(cell, (int, float)):
                        raise ValueError("Maze grid cells must be numbers.")

        return cls(
            size=data["size"],
            maze_number=data["maze_number"],
            grid=np.array(grid_data, dtype=int), # Ensure it's an int array
            generation_time=data.get("generation_time", 0.0)
        )

    def save(self, filename: str) -> None:
        """Save the maze to a JSON file."""
        data = {
            "size": self.size,
            "maze_number": self.maze_number,
            # Convert numpy array to list for JSON serialization
            "grid": self.grid.tolist() if isinstance(self.grid, np.ndarray) else self.grid,
            "generation_time": self.generation_time
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    # Added a basic string representation for easier debugging
    def __str__(self) -> str:
        return f"Mazes(size={self.size}, maze_number={self.maze_number}, grid_shape={self.grid.shape})" 