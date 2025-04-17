import json
import numpy as np

class Mazes:
    """
    Represents a single maze instance, storing its properties and
    providing methods for saving and loading.
    """
    def __init__(self, size: int, maze_number: int, grid: np.ndarray, generation_time: float):
        """
        Initializes a Mazes instance.

        Args:
            size (int): The dimension of the square maze (size x size).
            maze_number (int): An identifier for the maze within a set of the same size.
            grid (np.ndarray): A 2D numpy array representing the maze walls (1) and paths (0).
            generation_time (float): The time taken to generate the maze in seconds.
        """
        self.size = size
        self.maze_number = maze_number
        self.grid = grid
        self.generation_time = generation_time

    def save(self, filename: str):
        """
        Saves the maze instance to a file using JSON.

        Args:
            filename (str): The path to the file where the maze should be saved.
        """
        maze_data = {
            'size': self.size,
            'maze_number': self.maze_number,
            'grid': self.grid.tolist(),  # Convert numpy array to list
            'generation_time': self.generation_time
        }
        with open(filename, 'w') as f:
            json.dump(maze_data, f, indent=4)

    @classmethod
    def load(cls, filename: str):
        """
        Loads a maze instance from a JSON file.

        Args:
            filename (str): The path to the file from which to load the maze.

        Returns:
            Mazes: The loaded maze instance.
        """
        with open(filename, 'r') as f:
            maze_data = json.load(f)

        # Convert grid list back to numpy array
        grid_array = np.array(maze_data['grid'])

        return cls(
            size=maze_data['size'],
            maze_number=maze_data['maze_number'],
            grid=grid_array,
            generation_time=maze_data['generation_time']
        )

    def __str__(self) -> str:
        """Returns a string representation of the maze."""
        return f"Maze(size={self.size}x{self.size}, number={self.maze_number})"

    def __repr__(self) -> str:
        """Returns a detailed string representation for debugging."""
        return f"Mazes(size={self.size}, maze_number={self.maze_number}, grid=..., generation_time={self.generation_time:.4f})"

    def display(self):
        """Prints the maze grid to the console."""
        for row in self.grid:
            print("".join(["#" if cell == 1 else " " for cell in row]))
