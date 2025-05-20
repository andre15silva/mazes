import click
import os
import re
import json # For loading maze_filter_ids if provided as JSON string
from typing import List, Tuple, Optional, Dict, Any

# Solver engine components
from solver_engine.maze_utils import Mazes
from solver_engine.experiment_runner import ExperimentRunner
from solver_engine.results import save_results_to_json, print_summary_from_results, MazeSolution, load_results_from_json
from solver_engine.plotting_utils import plot_solution_to_file
from solver_engine.abc_solver import Solver

# Solvers
from solver_engine.traditional_solvers import MazelibCollisionSolver, MazelibBacktrackingSolver
from solver_engine.llm_solvers import (LLMSolverBase, OpenAISolver, 
                                   AsciiInputFormatter, EmojiInputFormatter, InputFormatter)

# Available solver classes mapping
AVAILABLE_TRADITIONAL_SOLVERS = {
    "collision": MazelibCollisionSolver,
    "backtracking": MazelibBacktrackingSolver,
}

AVAILABLE_LLM_INPUT_FORMATTERS = {
    "ascii": AsciiInputFormatter,
    "emoji": EmojiInputFormatter,
}

# Helper to parse maze IDs like "5x5_1,10x10_3"
def parse_maze_ids(ctx, param, value: Optional[str]) -> Optional[List[Tuple[int, int]]]:
    if not value:
        return None
    try:
        ids = []
        for item in value.split(','):
            match = re.match(r"(\d+)x\1_(\d+)", item.strip())
            if not match:
                raise click.BadParameter(f"Invalid maze ID format: '{item}'. Expected format like '5x5_1'.")
            size = int(match.group(1))
            num = int(match.group(2))
            ids.append((size, num))
        return ids
    except Exception as e:
        raise click.BadParameter(f"Error parsing maze IDs: {e}")

@click.command()
@click.option("--maze-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default="mazes", show_default=True, help="Directory containing maze JSON files.")
@click.option("--maze-filter-ids", type=str, callback=parse_maze_ids, help='Comma-separated list of maze IDs to process (e.g., "5x5_1,10x10_2"). Solves all if not specified.')
@click.option("--output-dir", type=click.Path(file_okay=False, dir_okay=True), default="outputs", show_default=True, help="Base directory to save results and plots.")
@click.option("--results-file", type=str, default="solver_results.json", show_default=True, help="Filename for the JSON results output, relative to output-dir.")

# Solver Type specific options
@click.option("--solver-type", type=click.Choice(['traditional', 'llm'], case_sensitive=False), required=True, help="Type of solver to use.")

# Traditional Solver Options
@click.option("--traditional-solver-name", type=click.Choice(list(AVAILABLE_TRADITIONAL_SOLVERS.keys()), case_sensitive=False), help="Name of the traditional solver to use (if solver-type is traditional).")

# LLM Solver Options
@click.option("--llm-model-name", type=str, default="gpt-3.5-turbo", show_default=True, help="LLM model name (e.g., gpt-4, gpt-3.5-turbo). Used if solver-type is llm.")
@click.option("--llm-input-format", type=click.Choice(list(AVAILABLE_LLM_INPUT_FORMATTERS.keys()), case_sensitive=False), default="ascii", show_default=True, help="Input format for LLM solvers.")
@click.option("--llm-max-trials", type=int, default=3, show_default=True, help="Maximum number of trials for LLM solvers.")

@click.option("--plot/--no-plot", default=True, show_default=True, help="Enable/disable saving of solution plots.")
@click.option("--plot-failed/--no-plot-failed", default=False, show_default=True, help="Also generate plots for failed/invalid solutions.")

def main(
    maze_dir: str,
    maze_filter_ids: Optional[List[Tuple[int, int]]],
    output_dir: str,
    results_file: str,
    solver_type: str,
    traditional_solver_name: Optional[str],
    llm_model_name: str,
    llm_input_format: str,
    llm_max_trials: int,
    plot: bool,
    plot_failed: bool,
):
    """Main CLI to run maze solving experiments."""
    os.makedirs(output_dir, exist_ok=True)
    plot_output_dir = os.path.join(output_dir, "plots")
    if plot:
        os.makedirs(plot_output_dir, exist_ok=True)

    # Load existing results if available
    results_json_path = os.path.join(output_dir, results_file)
    existing_results = []
    if os.path.exists(results_json_path):
        click.echo(f"Loading existing results from {results_json_path}")
        existing_results = load_results_from_json(results_json_path)

    # --- 1. Select and Prepare Maze Files ---
    all_maze_files_in_dir = []
    for fname in sorted(os.listdir(maze_dir)):
        if fname.endswith('.json'):
            match = re.match(r"maze_(\d+)x\1_(\d+)\.json", fname)
            if match:
                if maze_filter_ids:
                    size = int(match.group(1))
                    num = int(match.group(2))
                    if (size, num) in maze_filter_ids:
                        all_maze_files_in_dir.append(os.path.join(maze_dir, fname))
                else:
                    all_maze_files_in_dir.append(os.path.join(maze_dir, fname))
    
    if not all_maze_files_in_dir:
        click.echo(click.style(f"No maze files found or matched filter in '{maze_dir}'. Exiting.", fg="yellow"))
        return
    
    click.echo(f"Found {len(all_maze_files_in_dir)} maze files to process.")

    # --- 2. Initialize Solver(s) ---
    solvers_to_run: List[Solver] = []
    if solver_type == 'traditional':
        if not traditional_solver_name:
            click.echo(click.style("Error: --traditional-solver-name must be provided for solver-type 'traditional'.", fg="red"))
            return
        solver_class = AVAILABLE_TRADITIONAL_SOLVERS.get(traditional_solver_name)
        if not solver_class:
            click.echo(click.style(f"Error: Unknown traditional solver '{traditional_solver_name}'.", fg="red"))
            return
        solvers_to_run.append(solver_class())
    elif solver_type == 'llm':
        formatter_class = AVAILABLE_LLM_INPUT_FORMATTERS.get(llm_input_format)
        if not formatter_class:
            click.echo(click.style(f"Error: Unknown LLM input format '{llm_input_format}'.", fg="red"))
            return
        formatter: InputFormatter = formatter_class()
        try:
            llm_solver = OpenAISolver(
                formatter=formatter, 
                model_name=llm_model_name, 
                max_trials=llm_max_trials
            )
            solvers_to_run.append(llm_solver)
        except ValueError as e: # Handles API key error from OpenAISolver constructor
            click.echo(click.style(f"Error initializing LLM solver: {e}", fg="red"))
            return
    else:
        # Should not happen due to click.Choice
        click.echo(click.style(f"Error: Unknown solver type '{solver_type}'.", fg="red"))
        return

    if not solvers_to_run:
        click.echo(click.style("No solvers configured. Exiting.", fg="red"))
        return

    # --- 3. Run Experiments ---
    runner = ExperimentRunner(
        solvers=solvers_to_run, 
        maze_files=all_maze_files_in_dir,
        existing_results=existing_results
    )
    experiment_results = runner.run_experiments()

    # --- 4. Save Results ---
    save_results_to_json(experiment_results, results_json_path)

    # --- 5. Print Summary ---
    print_summary_from_results(experiment_results)

    # --- 6. Plotting ---
    if plot:
        click.echo(f"Generating plots in {plot_output_dir}...")
        plotted_count = 0
        for result in experiment_results:
            if result.valid or plot_failed:
                # Need the Mazes object again for plotting
                # This is a bit inefficient if we have many results; consider optimizing if it becomes slow.
                # For now, reload the maze for plotting.
                maze_obj_for_plot = None
                # Try to find the original maze file path from the result or re-construct
                # Assuming result.maze_id is (size, number)
                if isinstance(result.maze_id, (tuple, list)) and len(result.maze_id) == 2:
                    size, num = result.maze_id if isinstance(result.maze_id, tuple) else result.maze_id
                    expected_fname = f"maze_{size}x{size}_{num}.json"
                    for mfile in all_maze_files_in_dir: # Search within the list of files that were processed
                        if expected_fname == os.path.basename(mfile):
                            try:
                                maze_obj_for_plot = Mazes.load(mfile)
                            except Exception as e_load:
                                click.echo(f"Could not reload maze {mfile} for plotting: {e_load}")
                            break
                
                if not maze_obj_for_plot:
                    click.echo(f"Could not find/load maze for result: {result.solver_name} on {result.maze_id}. Skipping plot.")
                    continue

                # For LLM solvers, plot each attempt from history if requested and available
                if isinstance(solvers_to_run[0], LLMSolverBase) and 'llm_history' in result.metadata:
                    for attempt_idx, attempt_data in enumerate(result.metadata['llm_history']):
                        attempt_solution = MazeSolution(
                            solver_name=result.solver_name,
                            maze_id=result.maze_id,
                            path=attempt_data.get('path', []),
                            valid=not bool(attempt_data.get('invalid_first')), # Valid if no first_invalid index
                            solve_time=0, # Not relevant per attempt for plotting
                            metadata={}
                        )
                        # LLMSolverBase._validate_all gives all invalid indices for a path
                        invalid_indices_for_plot = attempt_data.get('invalid_all', [])
                        plot_solution_to_file(
                            maze_obj_for_plot, 
                            attempt_solution, 
                            plot_output_dir, 
                            attempt_number=attempt_idx + 1, # 1-indexed attempt
                            invalid_indices=invalid_indices_for_plot
                        )
                        plotted_count +=1
                else: # For traditional solvers or final LLM solution path
                    # For the final path from any solver, invalid_indices can be derived if needed, 
                    # or LLM's final solution.metadata might have it.
                    # Here, we pass None, so plot_solution_to_file uses result.valid for overall coloring.
                    plot_solution_to_file(maze_obj_for_plot, result, plot_output_dir, invalid_indices=result.metadata.get('invalid_all'))
                    plotted_count +=1
        click.echo(f"Generated {plotted_count} plots.")
    
    click.echo(click.style("All done!", fg="green"))

if __name__ == '__main__':
    main() 
