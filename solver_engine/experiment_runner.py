from typing import List, Callable, Dict, Any
from solver_engine.abc_solver import Solver
from solver_engine.maze_utils import Mazes
from solver_engine.results import MazeSolution
import time

class ExperimentRunner:
    """Runs a list of solvers over a list of maze files and collects results."""

    def __init__(
        self,
        solvers: List[Solver],
        maze_files: List[str],
        maze_loader: Callable[[str], Mazes] = Mazes.load
    ):
        if not solvers:
            raise ValueError("At least one solver must be provided.")
        if not maze_files:
            raise ValueError("At least one maze file must be provided.")
            
        self.solvers = solvers
        self.maze_files = maze_files
        self.maze_loader = maze_loader
        self.results: List[MazeSolution] = []

    def run_experiments(self, solver_kwargs: Dict[str, Any] = None) -> List[MazeSolution]:
        """
        Runs all configured solvers on all configured maze files.

        Args:
            solver_kwargs: Optional dictionary of keyword arguments to pass to each solver's
                           solve method. This can be used for solver-specific configurations
                           not handled at initialization (e.g. LLM temperature if not in constructor).
                           Currently, our LLMSolverBase takes most config at init (formatter, max_trials).

        Returns:
            A list of MazeSolution objects.
        """
        self.results = [] # Clear previous results if any
        if solver_kwargs is None:
            solver_kwargs = {}

        total_mazes = len(self.maze_files)
        total_solvers = len(self.solvers)
        print(f"Starting experiments: {total_mazes} mazes, {total_solvers} solvers each.")

        for i, maze_file in enumerate(self.maze_files):
            print(f"  Processing maze {i+1}/{total_mazes}: {maze_file}...")
            try:
                maze = self.maze_loader(maze_file)
            except Exception as e:
                print(f"    Error loading maze {maze_file}: {e}. Skipping this maze.")
                # Optionally, create a MazeSolution indicating loading error
                # For now, just skip.
                continue

            for j, solver_instance in enumerate(self.solvers):
                solver_display_name = getattr(solver_instance, 'solver_name', type(solver_instance).__name__)
                print(f"    Running solver {j+1}/{total_solvers}: {solver_display_name}...")
                try:
                    solution = solver_instance.solve(maze, **solver_kwargs)
                    self.results.append(solution)
                    valid_status = "SUCCESS" if solution.valid else "FAILURE"
                    print(f"      {solver_display_name} on {maze.maze_number}: {valid_status} in {solution.solve_time:.2f}s")
                except Exception as e:
                    print(f"    Error running solver {solver_display_name} on maze {maze.maze_number}: {e}")
                    # Create a MazeSolution indicating solver error
                    error_solution = MazeSolution(
                        solver_name=solver_display_name,
                        maze_id=(maze.size, maze.maze_number) if maze else maze_file, # use filename if maze failed to load
                        path=[],
                        valid=False,
                        solve_time=0.0, # Or measure time until error if meaningful
                        metadata={'error': str(e), 'details': 'Solver failed to execute'}
                    )
                    self.results.append(error_solution)
        
        print(f"Experiments finished. Total solutions collected: {len(self.results)}")
        return self.results 