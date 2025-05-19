from abc import ABC, abstractmethod
import time
import re
import ast
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from solver_engine.abc_solver import Solver
from solver_engine.maze_utils import Mazes
from solver_engine.results import MazeSolution
from solver_engine.llm_solvers.input_formatters import InputFormatter

class LLMSolverBase(Solver):
    """Abstract Base Class for LLM-based maze solvers."""

    def __init__(self, formatter: InputFormatter, max_trials: int = 5):
        self.formatter = formatter
        self.max_trials = max_trials

    @property
    @abstractmethod
    def solver_name(self) -> str:
        """The specific name of the LLM solver (e.g., OpenAISolver-gpt-4)."""
        pass

    @abstractmethod
    def _make_api_call(self, prompt: str) -> str:
        """Makes the actual API call to the LLM provider."""
        pass

    def _parse_response(self, response: str) -> List[Tuple[int, int]]:
        """Parses the LLM's string response into a list of (row, col) tuples."""
        # Adapted from llm_solver_script.py
        list_pat = r"\[(?:\s*\(\d+,\s*\d+\s*\)(?:,\s*\(\d+,\s*\d+\s*\))*)?\]"
        match = re.search(list_pat, response)
        if match:
            coords_str = match.group(0)
            # Handle cases like "[]" which ast.literal_eval might not like directly if not standard
            if coords_str == "[]":
                return []
            try:
                # More robust parsing of the matched list string
                parsed_list = ast.literal_eval(coords_str)
                if isinstance(parsed_list, list) and all(
                    isinstance(p, tuple) and len(p) == 2 and 
                    isinstance(p[0], int) and isinstance(p[1], int) 
                    for p in parsed_list
                ):
                    return parsed_list
            except (ValueError, SyntaxError):
                # Fallback to regex findall if literal_eval fails on the primary match
                pass 
        
        # Fallback or secondary attempt: find individual tuples if main list parsing fails or isn't found
        coords = re.findall(r"\((\d+),\s*(\d+)\)", response) # Find all (d,d) tuples
        if coords:
             return [(int(r), int(c)) for r, c in coords]

        # Try to find a list-like structure in any line for broader cases
        for line in response.splitlines():
            if '[' in line and ']' in line:
                try:
                    line_content = line[line.find('['):line.rfind(']')+1]
                    cand = ast.literal_eval(line_content)
                    if isinstance(cand, list) and all(isinstance(p, tuple) for p in cand):
                        # Basic check, could be more thorough like above
                        if all(len(p)==2 and isinstance(p[0], int) and isinstance(p[1], int) for p in cand):
                            return cand
                except (ValueError, SyntaxError):
                    pass
        return [] # Return empty list if no valid path found

    def _validate_first(self, maze: Mazes, path: List[Tuple[int, int]]) -> Tuple[bool, List[int]]:
        """Validates the path until the first error. Returns (isValid, [first_invalid_idx_or_empty])."""
        # Adapted from llm_solver_script.py
        dim = maze.grid.shape[0]
        # Standardized start/end points for our mazes based on typical border conventions
        start_point = (1, 0)
        end_point = (dim - 2, dim - 1)

        if not path or path[0] != start_point:
            return False, [0]
        
        # Check for self-intersections (revisiting cells)
        if len(path) > len(set(path)):
             # Find first revisited cell to mark as invalid for feedback
            visited_so_far = set()
            for idx, p_cell in enumerate(path):
                if p_cell in visited_so_far:
                    return False, [idx]
                visited_so_far.add(p_cell)

        for idx, current_pos in enumerate(path):
            r, c = current_pos
            if not (0 <= r < dim and 0 <= c < dim and maze.grid[r, c] == 0):
                return False, [idx] # Wall, out of bounds
            if idx > 0:
                prev_pos = path[idx-1]
                r_prev, c_prev = prev_pos
                if abs(r - r_prev) + abs(c - c_prev) != 1:
                    return False, [idx] # Invalid move (diagonal, too far, or same cell)
        
        if not path or path[-1] != end_point:
            # If path is empty, this won't be hit due to first check. If non-empty & last not end.
            return False, [len(path) -1 if path else 0]
            
        return True, []

    def _validate_all(self, maze: Mazes, path: List[Tuple[int, int]]) -> List[int]:
        """Validates the entire path and returns all invalid indices."""
        # Adapted from llm_solver_script.py
        dim = maze.grid.shape[0]
        start_point = (1, 0)
        end_point = (dim - 2, dim - 1)
        invalid_indices = []

        if not path or path[0] != start_point:
            invalid_indices.append(0)
        
        # Check for self-intersections (revisiting cells)
        # This check is complex for _validate_all as we need to mark all segments involved
        # For simplicity, the original _validate_all did not explicitly check for self-intersection
        # in a way that returns all involved indices. We'll stick to its original behavior.
        # A full self-intersection check for *all* invalid segments is harder.
        # The _validate_first handles the first occurrence, which is used for feedback.

        for idx, current_pos in enumerate(path):
            r, c = current_pos
            is_segment_valid = True
            if not (0 <= r < dim and 0 <= c < dim and maze.grid[r, c] == 0):
                is_segment_valid = False # Wall, out of bounds
            
            if idx > 0:
                prev_pos = path[idx-1]
                r_prev, c_prev = prev_pos
                if abs(r - r_prev) + abs(c - c_prev) != 1:
                    is_segment_valid = False # Invalid move
            elif idx == 0 and current_pos != start_point: # Redundant with first check but for clarity
                is_segment_valid = False

            if not is_segment_valid and idx not in invalid_indices:
                 invalid_indices.append(idx)

        if path and path[-1] != end_point:
            # Mark the last segment if it doesn't reach the end
            last_idx = len(path) - 1
            if last_idx not in invalid_indices:
                 invalid_indices.append(last_idx)
        elif not path and 0 not in invalid_indices: # If path is empty and 0 wasn't added
             invalid_indices.append(0)

        # Add check for revisited cells, marking the *second* occurrence onwards
        visited_so_far = set()
        for idx, p_cell in enumerate(path):
            if p_cell in visited_so_far:
                if idx not in invalid_indices:
                    invalid_indices.append(idx)
            else:
                visited_so_far.add(p_cell)
        
        return sorted(list(set(invalid_indices))) # Unique sorted indices

    def _format_prompt(self, maze: Mazes, history: List[Dict[str, Any]]) -> str:
        """Formats the complete prompt for the LLM, including history."""
        dim = maze.grid.shape[0]
        # Standardized start/end points
        start_point = (1, 0)
        end_point = (dim - 2, dim - 1)

        maze_map_str = self.formatter.format_grid(maze.grid)
        wall_char = self.formatter.wall_char
        path_char = self.formatter.path_char

        prompt_lines = [
            "You are given a square maze, represented by a 2D grid of characters:",
            f"  - Walls: {wall_char} (impassable)",
            f"  - Free cells: {path_char} (traversable)",
            "",
            f"Coordinates are 0-indexed: (0,0) is the top-left, ({dim-1},{dim-1}) is the bottom-right.",
            f"Start position: {start_point}",
            f"End position:   {end_point}",
            "",
            "Rules:",
            "  1. You may move one cell at a time: up, down, left, or right.",
            "  2. No diagonal moves.",
            f"  3. No jumping over {wall_char} cells.",
            f"  4. You cannot move into {wall_char} cells.",
            "  5. You cannot revisit any cell (including the start and end points if part of a longer path attempt).",
            "",
            "Output:",
            "  - A single Python list of (row, col) tuples, in order from the start to the end, including both endpoints.",
            "  - The entire list must appear on one line, with no extra text or commentary.",
            "",
            "Here is the maze map (each line is one row):",
            maze_map_str
        ]

        for i, attempt_record in enumerate(history, 1):
            attempt_path = attempt_record['path']
            first_invalid_indices = attempt_record['invalid_first'] # This is a list
            
            prompt_lines.append("")
            if not first_invalid_indices: # Previous attempt was valid but perhaps not submitted or was a partial success
                prompt_lines.append(f"Here was a previous valid path segment attempt #{i}:\n{attempt_path}")
                prompt_lines.append("This path was valid up to the point shown. Continue or restart if necessary.")
            else:
                first_invalid_idx = first_invalid_indices[0]
                # Display path up to and including the first invalid step
                path_segment_to_show = attempt_path[:first_invalid_idx + 1]
                prompt_lines.append(f"Here is a previous attempt (Attempt #{i}) including all steps until the first invalid step ({attempt_path[first_invalid_idx]}) at index {first_invalid_idx}:\n{path_segment_to_show}")
                prompt_lines.append("The last point shown was where the path became invalid. Do not repeat this mistake. You may need to backtrack from earlier valid steps. Do not assume all other steps before the error are correct parts of the final solution.")

        return "\n".join(prompt_lines)

    def solve(self, maze: Mazes, **kwargs) -> MazeSolution:
        """Solves the maze using an LLM, with retries and feedback."""
        start_solve_time = time.time()
        
        history: List[Dict[str, Any]] = []
        final_path: List[Tuple[int, int]] = []
        is_valid_solution = False
        raw_llm_response = ""

        for trial in range(1, self.max_trials + 1):
            current_prompt = self._format_prompt(maze, history)
            
            # Optional: print prompt for debugging
            # print(f"--- TRIAL {trial} PROMPT ---")
            # print(current_prompt)
            # print("-------------------------")

            raw_llm_response = self._make_api_call(current_prompt)
            
            # Optional: print raw response for debugging
            # print(f"--- TRIAL {trial} RAW RESPONSE ---")
            # print(raw_llm_response)
            # print("-----------------------------")

            parsed_path = self._parse_response(raw_llm_response)
            
            # Optional: print parsed path for debugging
            # print(f"--- TRIAL {trial} PARSED PATH ---")
            # print(parsed_path)
            # print("-----------------------------")

            is_path_ok, first_invalid = self._validate_first(maze, parsed_path)
            all_invalid = self._validate_all(maze, parsed_path)

            history.append({
                'prompt': current_prompt, # For potential later analysis
                'raw_response': raw_llm_response,
                'path': parsed_path,
                'invalid_first': first_invalid, 
                'invalid_all': all_invalid,
                'trial_number': trial
            })

            if is_path_ok:
                final_path = parsed_path
                is_valid_solution = True
                break # Successful solution found
        
        total_solve_time = time.time() - start_solve_time

        return MazeSolution(
            solver_name=self.solver_name,
            maze_id=(maze.size, maze.maze_number),
            path=final_path,
            valid=is_valid_solution,
            solve_time=total_solve_time,
            metadata={
                'llm_raw_response': raw_llm_response,
                'llm_history': history,
                'trials_taken': len(history)
            }
        ) 