# Maze Generation and Solving

This project generates mazes of various sizes using Prim's algorithm and solves them using both traditional algorithms and LLM-based solvers. It measures the time taken for both generation and solving steps, and supports experiment management, result saving, and plotting.

## Project Structure

```
.
├── generate_mazes.py         # Script to generate mazes and save them as JSON files
├── main_solver.py            # Main CLI for running maze solving experiments (traditional & LLM)
├── mazes.py                  # Contains the Mazes class for maze data handling
├── mazes/                    # Directory where generated maze JSON files are stored
├── outputs/                  # Output directory for results and plots (created by main_solver.py)
├── solver_engine/            # Core experiment, solver, and plotting logic
│   ├── experiment_runner.py
│   ├── results.py
│   ├── plotting_utils.py
│   ├── maze_utils.py
│   ├── abc_solver.py
│   ├── llm_solvers/
│   └── traditional_solvers/
├── pyproject.toml            # Project dependencies and configuration
└── README.md                 # This file
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

4.  **(Optional, for LLM solvers)**: Set your OpenAI API key as an environment variable:
    ```bash
    export OPENAI_API_KEY=your-key-here
    ```
    Or add it to a `.env` file in the project root.

## Usage

### 1. Generate Mazes
Run the generation script to create the `mazes/` directory and populate it with JSON files, each containing data for one maze instance:
```bash
python generate_mazes.py
```
You will see output indicating the progress of maze generation and saving.

### 2. Solve Mazes (Main Entry Point)
Use `main_solver.py` to solve mazes using either traditional or LLM-based solvers. This script loads maze data from the `mazes/` directory, runs the selected solver(s), and saves results and plots in the `outputs/` directory.

#### Example usage:
```bash
python main_solver.py --solver-type traditional --traditional-solver-name collision
```

#### CLI Options (partial list):
- `--maze-dir`: Directory containing maze JSON files (default: `mazes`)
- `--maze-filter-ids`: Comma-separated list of maze IDs to process (e.g., `5x5_1,10x10_2`). Solves all if not specified.
- `--output-dir`: Directory to save results and plots (default: `outputs`)
- `--results-file`: Filename for the JSON results output (default: `solver_results.json`)
- `--solver-type`: `traditional` or `llm` (**required**)
- `--traditional-solver-name`: Name of the traditional solver to use (if `solver-type` is `traditional`)
- `--llm-model-name`: LLM model name (e.g., `gpt-4`, `gpt-3.5-turbo`) (if `solver-type` is `llm`)
- `--llm-input-format`: Input format for LLM solvers (`ascii` or `emoji`)
- `--llm-max-trials`: Maximum number of trials for LLM solvers
- `--plot/--no-plot`: Enable/disable saving of solution plots (default: enabled)
- `--plot-failed/--no-plot-failed`: Also generate plots for failed/invalid solutions

Run with `--help` for the full list of options:
```bash
python main_solver.py --help
```

#### Output
- Results are saved as a JSON file (default: `outputs/solver_results.json`).
- Plots of solutions are saved in `outputs/plots/`.
- A summary of results is printed to the console.

## Available Solvers

### Traditional Solvers
- `collision`: Uses the `mazelib` Collision algorithm
- `backtracking`: Uses the `mazelib` Backtracking algorithm

### LLM Solvers
- `OpenAISolver`: Uses the OpenAI API (requires API key)
    - `--llm-model-name` (e.g., `gpt-4.1`, `o4-mini`)
    - `--llm-input-format`:
        - `ascii`: Maze as ASCII art (`#` for wall, `.` for path)
        - `emoji`: Maze as emoji (`❤️` for wall, `🙂` for path)

## Results and Plotting
- Results are stored in JSON format, including solver name, maze ID, path, validity, solve time, and metadata.
- Plots are generated for each solution (and optionally for failed attempts).
- Summaries include success rates and average solve times per solver.

## Experiment Comparison

The `experiment_comparison.py` script allows you to compare results from multiple experiment runs (i.e., different output directories containing `solver_results.json`). It generates summary statistics and a variety of comparison plots (solve times, success rates, trial progression) across solvers and maze sizes.

### Example usage:
```bash
python experiment_comparison.py outputs1 outputs2 --output-dir comparison_plots
```
- `outputs1`, `outputs2`, ...: One or more directories containing `solver_results.json` files (as produced by `main_solver.py`).
- `--output-dir`: Directory to save the generated comparison plots and summary statistics (default: `comparison_plots`).

The script will generate:
- CSV file with summary statistics (`summary_stats.csv`)
- Plots comparing solve times and success rates by maze size and solver
- (For LLM solvers) Plots showing trial progression and first-trial vs. final success rates

This is useful for benchmarking and visualizing the performance of different solvers or experiment configurations.

## The `Mazes` Class (`mazes.py`)

The `Mazes` class provides a standardized way to store maze properties (size, grid, generation time) and handle saving/loading this data to/from disk.

- `save(filename)`: Saves the maze instance to a JSON file.
- `load(filename)`: Loads maze data from a JSON file and creates a `Mazes` instance.
- `display()`: Prints a simple text representation of the maze grid to the console.

This class ensures that maze data is consistently handled between the generation and solving scripts. 