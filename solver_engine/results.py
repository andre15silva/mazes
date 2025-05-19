from dataclasses import dataclass, field
from typing import List, Tuple, Any, Dict
import json # Added for save_results_to_json

@dataclass
class MazeSolution:
    solver_name: str
    maze_id: Any # e.g., tuple of (dim, number) or filename
    path: List[Tuple[int, int]]
    valid: bool
    solve_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "solver_name": self.solver_name,
            "maze_id": self.maze_id,
            "path": self.path,
            "valid": self.valid,
            "solve_time": self.solve_time,
            "metadata": self.metadata,
        }

def save_results_to_json(results: List[MazeSolution], output_filepath: str) -> None:
    """Saves a list of MazeSolution objects to a JSON file."""
    try:
        data_to_save = [res.to_dict() for res in results]
        with open(output_filepath, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        print(f"Results successfully saved to {output_filepath}")
    except IOError as e:
        print(f"Error saving results to {output_filepath}: {e}")
    except TypeError as e:
        print(f"Error serializing results to JSON: {e}. Check metadata contents.")

def load_results_from_json(input_filepath: str) -> List[MazeSolution]:
    """Loads a list of MazeSolution objects from a JSON file."""
    results = []
    try:
        with open(input_filepath, 'r') as f:
            data_loaded = json.load(f)
        for item in data_loaded:
            # Basic reconstruction. More complex metadata might need custom handling.
            results.append(MazeSolution(**item))
        print(f"Results successfully loaded from {input_filepath}")
    except FileNotFoundError:
        print(f"Error: Results file not found at {input_filepath}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {input_filepath}: {e}")
    except TypeError as e:
        print(f"Error creating MazeSolution from loaded data: {e}. Check file structure.")
    return results

def print_summary_from_results(results: List[MazeSolution]):
    """Prints a summary of success rates and average times per solver."""
    if not results:
        print("No results to summarize.")
        return

    stats: Dict[str, Dict[str, Any]] = {}
    for r in results:
        # Ensure maze_id is hashable for dictionary keys if it's a tuple (dim, number)
        # If maze_id can be other non-hashable types, this part needs adjustment
        # For now, assume solver_name is the primary key for aggregation.
        solver_stats = stats.setdefault(r.solver_name, {"total": 0, "successes": 0, "total_time": 0.0, "paths_found": 0})
        solver_stats['total'] += 1
        if r.valid:
            solver_stats['successes'] += 1
        if r.path: # Count if a path was returned, even if not ultimately valid by all criteria
            solver_stats['paths_found'] +=1
        solver_stats['total_time'] += r.solve_time
    
    print("\n--- Experiment Summary ---")
    for solver_name, data in stats.items():
        success_rate = (data['successes'] / data['total'] * 100) if data['total'] > 0 else 0
        avg_time = (data['total_time'] / data['total']) if data['total'] > 0 else 0
        print(f"Solver: {solver_name}")
        print(f"  Total Mazes Attempted: {data['total']}")
        print(f"  Successful Solutions (valid path): {data['successes']} ({success_rate:.2f}%)")
        print(f"  Paths Returned (any, valid or not): {data['paths_found']}")
        print(f"  Average Solve Time: {avg_time:.3f}s")
        print("------------------------") 