from mazelib import Maze
from mazelib.solve.Collision import Collision
from mazelib.solve.RandomMouse import RandomMouse
from mazelib.solve.BacktrackingSolver import BacktrackingSolver

import pandas as pd
import os
import time
from mazes import Mazes

def solve_maze(maze_grid, solver, start_point, end_point):
    """Solve a maze using the specified solver."""
    m = Maze()
    m.grid = maze_grid
    # Set start and end points explicitly based on common mazelib conventions
    m.start = start_point
    m.end = end_point
    m.solver = solver
    # m.generate_entrances() # No longer needed as we set start/end
    m.solve()
    return m.solutions

def main():
    input_dir = 'mazes' # Directory where maze JSON files are stored
    # Load maze file paths
    print(f"Loading mazes from '{input_dir}' directory...")
    try:
        maze_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.json')]
    except FileNotFoundError:
        print(f"Error: Directory '{input_dir}' not found. Please run generate_mazes.py first.")
        return

    if not maze_files:
        print(f"No maze JSON files found in '{input_dir}'.")
        return

    # Initialize solvers
    solvers = {
        # 'Backtracking': BacktrackingSolver(),
        'Collision': Collision(),
        # 'RandomMouse': RandomMouse()
    }

    results = []

    print("\nSolving mazes...")
    for maze_file in maze_files:
        # Load the Mazes instance from JSON
        maze_instance = Mazes.load(maze_file)
        size = maze_instance.size
        maze_number = maze_instance.maze_number
        maze_grid = maze_instance.grid # Use the grid directly

        print(f"\nSolving {maze_file} (Size: {size}x{size}, Number: {maze_number})")

        # Define start and end points (assuming top-left and bottom-right access points)
        # Adjust if your generation logic creates different entrances/exits
        start_point = (1, 0) # Corresponds to grid[1][0] being open
        end_point = (maze_grid.shape[0] - 2, maze_grid.shape[1] - 1) # Corresponds to grid[end_row][end_col] being open

        # Try each solver
        for solver_name, solver_instance in solvers.items():
            start_time = time.time()
            try:
                # Pass the grid and solver instance
                solutions = solve_maze(maze_grid, solver_instance, start_point, end_point)
                solve_time = time.time() - start_time

                solution_found = bool(solutions and solutions[0])
                solution_length = len(solutions[0]) if solution_found else 0

                # Store results using data from the maze_instance
                result = {
                    'size': size,
                    'maze_number': maze_number,
                    'solver': solver_name,
                    'solve_time': solve_time,
                    'solution_found': solution_found,
                    'solution_length': solution_length
                }
                results.append(result)

                print(f"{solver_name}: {'Solved' if result['solution_found'] else 'No solution'} "
                      f"in {solve_time:.4f}s (Length: {solution_length})")

            except Exception as e:
                print(f"{solver_name}: Error - {str(e)}")
                results.append({
                    'size': size,
                    'maze_number': maze_number,
                    'solver': solver_name,
                    'solve_time': -1,
                    'solution_found': False,
                    'solution_length': 0
                })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    output_file = 'maze_solutions.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # Print summary statistics
    print("\nSummary Statistics:")
    # Filter out errors for summary stats
    valid_results = results_df[results_df['solve_time'] >= 0]
    if not valid_results.empty:
        summary = valid_results.groupby(['size', 'solver']).agg(
            mean_solve_time = pd.NamedAgg(column='solve_time', aggfunc='mean'),
            std_solve_time = pd.NamedAgg(column='solve_time', aggfunc='std'),
            success_rate = pd.NamedAgg(column='solution_found', aggfunc='mean'),
            mean_solution_length = pd.NamedAgg(column='solution_length', aggfunc=lambda x: x[x>0].mean()) # Mean length only for solved
        ).round(4)
        # Fill NaN in std dev (for single-entry groups) and mean length (if no solutions found for a group)
        summary = summary.fillna({'std_solve_time': 0, 'mean_solution_length': 0})
        print(summary)
    else:
        print("No valid results to summarize.")


if __name__ == "__main__":
    main()