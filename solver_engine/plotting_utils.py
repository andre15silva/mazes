import os
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np # Will be needed for grid type hints and operations
from typing import List, Tuple, Optional # For type hints

from solver_engine.maze_utils import Mazes # For Mazes type hint
from solver_engine.results import MazeSolution # For MazeSolution type hint

# Define a common color map
# 0: path (white), 1: wall (black)
# For plotting paths:
# Green: valid path segment, Red: invalid path segment
DEFAULT_CMAP = colors.ListedColormap(['white', 'black'])
VALID_PATH_COLOR = 'green'
INVALID_PATH_COLOR = 'red'

def plot_solution_to_file(
    maze: Mazes,
    solution: MazeSolution,
    output_dir: str,
    attempt_number: Optional[int] = None, # For LLM history items, if plotting individual attempts
    invalid_indices: Optional[List[int]] = None # Explicitly pass invalid indices for a given path
) -> Optional[str]:
    """
    Plots a given maze solution (or a path from it) and saves it to a file.

    Args:
        maze: The Mazes object.
        solution: The MazeSolution object. The path from this solution will be plotted.
                  If `invalid_indices` is not provided, it's assumed this is the final path
                  and its `valid` status determines coloring (though it will be all green if valid).
        output_dir: Base directory to save the plot.
        attempt_number: Optional. If provided, used in filename for plotting specific attempts (e.g., from LLM history).
        invalid_indices: Optional list of indices in `solution.path` that are invalid.
                         If None, and solution.valid is False, it implies the whole path might be considered
                         problematic or the reason for invalidity is not segment-specific for plotting.
                         If solution.valid is True and this is None, all segments are green.

    Returns:
        The path to the saved plot file, or None if plotting failed.
    """
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            return None

    path_to_plot = solution.path
    if not path_to_plot:
        # print(f"No path data in solution for {solution.solver_name} on maze {solution.maze_id}. Skipping plot.")
        # Still, we might want to plot the empty maze in this case.
        pass # Continue to plot the maze even if path is empty

    dim = maze.grid.shape[0]
    fig, ax = plt.subplots(figsize=(max(5, dim / 2), max(5, dim / 2))) # Adjust size with maze dim
    ax.imshow(maze.grid, cmap=DEFAULT_CMAP, origin='upper', interpolation='nearest')

    plot_title = f"{solution.solver_name} - Maze {solution.maze_id}"
    if attempt_number is not None:
        plot_title += f" - Attempt {attempt_number}"
    status_suffix = " (Successful)" if solution.valid else " (Failed/Partial)"
    if not path_to_plot and not solution.valid:
        status_suffix = " (No Path Found)"
    elif not path_to_plot and solution.valid: # Should not happen if valid means a full path
        status_suffix = " (Valid but No Path?)" 

    ax.set_title(plot_title + status_suffix, fontsize=10)
    ax.set_xticks(np.arange(-.5, dim, 1), minor=True)
    ax.set_yticks(np.arange(-.5, dim, 1), minor=True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, which='minor', color='k', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)
    #ax.set_xticks(range(dim)); ax.set_yticks(range(dim)); ax.grid(True, linewidth=0.5)

    for i, (r, c) in enumerate(path_to_plot):
        color = VALID_PATH_COLOR
        if invalid_indices and i in invalid_indices:
            color = INVALID_PATH_COLOR
        elif not solution.valid and not invalid_indices: # If path failed as a whole, mark all red
            # This behavior might need refinement. If invalid_indices is the source of truth for bad segments,
            # this 'else' might be too broad if a path is invalid for reasons other than specific bad segments
            # (e.g. not reaching the end, but all segments are individually fine).
            # For now, if solution is invalid and no specific indices given, assume whole path is suspect.
            if not solution.path : # if path is empty and solution invalid
                 pass # no path to color
            else:
                color = INVALID_PATH_COLOR 

        ax.scatter(c, r, c=color, s=80, marker='s', edgecolors='none', alpha=0.7)
        ax.text(c, r, str(i), va='center', ha='center', color='black', fontsize=max(6, 12 - dim // 5))
    
    # Mark standard start and end points on the maze visually
    std_start_r, std_start_c = 1, 0
    std_end_r, std_end_c = dim - 2, dim - 1
    ax.plot(std_start_c, std_start_r, 'go', markersize=10, alpha=0.5, label='Std. Start') # Green circle
    ax.plot(std_end_c, std_end_r, 'ro', markersize=10, alpha=0.5, label='Std. End')   # Red circle

    ax.invert_yaxis() # Match array indexing (0,0 at top-left)

    # Construct filename
    maze_id_str = f"maze_{maze.size}x{maze.size}_{maze.maze_number}"
    solver_name_fs = solution.solver_name.replace(" ", "_").replace(":", "-")
    attempt_suffix = f"_attempt_{attempt_number}" if attempt_number is not None else "_final"
    
    # Ensure output_dir includes solver name and maze_id for organization
    # Example: output_dir_base/SolverName/MazeID/plot.pdf
    final_plot_dir = os.path.join(output_dir, solver_name_fs, maze_id_str)
    os.makedirs(final_plot_dir, exist_ok=True)

    plot_filename = f"{maze_id_str}_{solver_name_fs}{attempt_suffix}.pdf"
    full_plot_path = os.path.join(final_plot_dir, plot_filename)

    try:
        plt.savefig(full_plot_path, bbox_inches='tight')
        # print(f"Plot saved to {full_plot_path}")
    except Exception as e:
        print(f"Error saving plot to {full_plot_path}: {e}")
        plt.close(fig)
        return None
    
    plt.close(fig) # Close the figure to free memory
    return full_plot_path 

def plot_empty_maze_to_file(
    maze: Mazes,
    output_dir: str
) -> Optional[str]:
    """
    Plots the empty maze grid (no path overlay) and saves it to a file.

    Args:
        maze: The Mazes object.
        output_dir: Directory to save the plot.

    Returns:
        The path to the saved plot file, or None if plotting failed.
    """
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            return None

    dim = maze.grid.shape[0]
    fig, ax = plt.subplots(figsize=(max(5, dim / 2), max(5, dim / 2)))
    ax.imshow(maze.grid, cmap=DEFAULT_CMAP, origin='upper', interpolation='nearest')

    plot_title = f"Empty Maze - Maze {maze.size}x{maze.size}_{maze.maze_number}"
    ax.set_title(plot_title, fontsize=10)
    ax.set_xticks(np.arange(-.5, dim, 1), minor=True)
    ax.set_yticks(np.arange(-.5, dim, 1), minor=True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, which='minor', color='k', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)
    ax.invert_yaxis()

    # Mark standard start and end points
    std_start_r, std_start_c = 1, 0
    std_end_r, std_end_c = dim - 2, dim - 1
    ax.plot(std_start_c, std_start_r, 'go', markersize=10, alpha=0.5, label='Std. Start')
    ax.plot(std_end_c, std_end_r, 'ro', markersize=10, alpha=0.5, label='Std. End')

    # Construct filename
    maze_id_str = f"maze_{maze.size}x{maze.size}_{maze.maze_number}"
    final_plot_dir = os.path.join(output_dir, "empty_maze", maze_id_str)
    os.makedirs(final_plot_dir, exist_ok=True)
    plot_filename = f"{maze_id_str}_empty.pdf"
    full_plot_path = os.path.join(final_plot_dir, plot_filename)

    try:
        plt.savefig(full_plot_path, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving plot to {full_plot_path}: {e}")
        plt.close(fig)
        return None
    plt.close(fig)
    return full_plot_path 