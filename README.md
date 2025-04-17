# Maze Generation and Solving

This project generates mazes of various sizes using Prim's algorithm and then solves them using different algorithms provided by the `mazelib` library. It measures the time taken for both generation and solving steps.

## Project Structure

```
.
├── generate_mazes.py  # Script to generate mazes and save them as JSON files
├── solve_mazes.py     # Script to load mazes from JSON and solve them
├── mazes.py           # Contains the Mazes class for maze data handling
├── mazes/             # Directory where generated maze JSON files are stored (created by generate_mazes.py)
├── maze_solutions.csv # Output CSV file with solving results (created by solve_mazes.py)
├── pyproject.toml     # Project dependencies and configuration
└── README.md          # This file
```

## Setup

This project uses `uv` for environment and dependency management.

1.  **Install uv:** If you don't have `uv` installed, follow the instructions on the [official uv website](https://github.com/astral-sh/uv).

2.  **Create a virtual environment and install dependencies:**
    ```bash
    uv venv
    uv sync
    ```
    This will create a virtual environment (usually named `.venv`) and install the packages listed in `pyproject.toml`.

3.  **Activate the virtual environment:**
    *   On Linux/macOS:
        ```bash
        source .venv/bin/activate
        ```
    *   On Windows (PowerShell):
        ```powershell
        .venv\Scripts\Activate.ps1
        ```
    *   On Windows (CMD):
        ```cmd
        .venv\Scripts\activate.bat
        ```

## Usage

Make sure your virtual environment is activated before running the scripts.

1.  **Generate Mazes:**
    Run the generation script. This will create the `mazes/` directory (if it doesn't exist) and populate it with JSON files, each containing data for one maze instance.
    ```bash
    python generate_mazes.py
    ```
    You will see output indicating the progress of maze generation and saving.

2.  **Solve Mazes:**
    After generating the mazes, run the solving script. This script loads the maze data from the JSON files in the `mazes/` directory, attempts to solve each maze using the configured algorithms, and saves the results (solver used, time taken, solution found, solution length) to `maze_solutions.csv`.
    ```bash
    python solve_mazes.py
    ```
    You will see output showing the progress of solving each maze and the final summary statistics printed to the console. The results are also saved in `maze_solutions.csv`.

## The `Mazes` Class (`mazes.py`)

The `mazes.py` file defines the `Mazes` class, which acts as a container for all information related to a single maze instance.

*   **Purpose:** To provide a standardized way to store maze properties (size, grid, generation time) and handle saving/loading this data to/from disk.
*   **Attributes:**
    *   `size`: The logical dimension of the maze (e.g., 5 for a 5x5 maze).
    *   `maze_number`: An identifier for the maze within a set of the same size.
    *   `grid`: A NumPy array representing the maze structure (walls and paths).
    *   `generation_time`: Time taken to generate the maze.
*   **Methods:**
    *   `save(filename)`: Saves the maze instance data to a specified JSON file.
    *   `load(filename)`: A class method to load maze data from a JSON file and create a `Mazes` instance.
    *   `display()`: Prints a simple text representation of the maze grid to the console.

Using this class ensures that maze data is consistently handled between the generation and solving scripts. 