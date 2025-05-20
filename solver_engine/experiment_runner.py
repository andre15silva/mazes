from typing import List, Callable, Dict, Any, Set, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from solver_engine.abc_solver import Solver
from solver_engine.maze_utils import Mazes
from solver_engine.results import MazeSolution
import time
import os

class ExperimentRunner:
    """Runs a list of solvers over a list of maze files and collects results."""

    def __init__(
        self,
        solvers: List[Solver],
        maze_files: List[str],
        maze_loader: Callable[[str], Mazes] = Mazes.load,
        existing_results: List[MazeSolution] = None
    ):
        if not solvers:
            raise ValueError("At least one solver must be provided.")
        if not maze_files:
            raise ValueError("At least one maze file must be provided.")
            
        self.solvers = solvers
        self.maze_files = maze_files
        self.maze_loader = maze_loader
        self.existing_results = existing_results or []
        self.results: List[MazeSolution] = []

    def _get_attempted_combinations(self) -> Set[Tuple[str, Any]]:
        """Returns a set of (solver_name, maze_id) tuples that have already been attempted."""
        attempted = set()
        for result in self.existing_results:
            attempted.add((result.solver_name, tuple(result.maze_id) if isinstance(result.maze_id, list) else result.maze_id))
        return attempted

    def run_experiments(self, solver_kwargs: Dict[str, Any] = None, max_workers: int = None) -> List[MazeSolution]:
        """
        Runs all configured solvers on all maze files in parallel, skipping combinations that have already been attempted.

        Args:
            solver_kwargs: Optional dictionary of keyword arguments to pass to each solver's
                           solve method.
            max_workers: Maximum number of worker threads to use. If None, will use default
                        ThreadPoolExecutor behavior (typically min(32, os.cpu_count() + 4)).

        Returns:
            A list of MazeSolution objects, including both existing and new results.
        """
        self.results = [] # Clear previous results if any
        if solver_kwargs is None:
            solver_kwargs = {}

        # Get already attempted combinations
        attempted_combinations = self._get_attempted_combinations()

        total_mazes = len(self.maze_files)
        total_solvers = len(self.solvers)
        print(f"Starting parallel experiments: {total_mazes} mazes, {total_solvers} solvers each.")
        if self.existing_results:
            print(f"Found {len(self.existing_results)} existing results that will be preserved.")

        def process_maze_solver(maze_file: str, solver_instance: Solver) -> Optional[MazeSolution]:
            solver_display_name = getattr(solver_instance, 'solver_name', type(solver_instance).__name__)
            try:
                maze = self.maze_loader(maze_file)
                # Check if this combination has already been attempted
                maze_id = (maze.size, maze.maze_number)
                if (solver_display_name, maze_id) in attempted_combinations:
                    print(f"      Skipping {solver_display_name} on {maze.maze_number} (already attempted)")
                    return None
            except Exception as e:
                print(f"    Error loading maze {maze_file}: {e}")
                return MazeSolution(
                    solver_name=solver_display_name,
                    maze_id=maze_file,
                    path=[],
                    valid=False,
                    solve_time=0.0,
                    metadata={'error': str(e), 'details': 'Maze failed to load'}
                )

            try:
                solution = solver_instance.solve(maze, **solver_kwargs)
                valid_status = "SUCCESS" if solution.valid else "FAILURE"
                print(f"      {solver_display_name} on {maze.maze_number}: {valid_status} in {solution.solve_time:.2f}s")
                return solution
            except Exception as e:
                print(f"    Error running solver {solver_display_name} on maze {maze.maze_number}: {e}")
                return MazeSolution(
                    solver_name=solver_display_name,
                    maze_id=(maze.size, maze.maze_number) if maze else maze_file,
                    path=[],
                    valid=False,
                    solve_time=0.0,
                    metadata={'error': str(e), 'details': 'Solver failed to execute'}
                )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create all combinations of maze files and solvers
            futures = []
            for maze_file in self.maze_files:
                for solver_instance in self.solvers:
                    futures.append(
                        executor.submit(process_maze_solver, maze_file, solver_instance)
                    )
            
            # Collect results as they complete
            for future in as_completed(futures):
                solution = future.result()
                if solution is not None:  # Only add new solutions
                    self.results.append(solution)

        # Combine existing and new results
        combined_results = self.existing_results + self.results
        print(f"Parallel experiments finished. Total solutions: {len(combined_results)} ({len(self.results)} new)")
        return combined_results 