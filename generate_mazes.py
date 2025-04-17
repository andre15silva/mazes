from mazelib import Maze
from mazelib.generate.Prims import Prims

import time
import os
from mazes import Mazes

def generate_maze(size):
    """Generate a maze of given size using Prim's algorithm."""
    m = Maze()
    # Use 2*size + 1 for grid dimensions to match mazelib's internal representation
    actual_dim = size # Keep size as the logical size
    m.generator = Prims(actual_dim, actual_dim)
    m.generate()
    # Ensure start and end points are accessible (optional, but good practice)
    m.grid[1, 0] = 0
    m.grid[actual_dim*2-1, actual_dim*2] = 0
    return m.grid

def main():
    sizes = [5, 10, 25, 50, 100]
    output_dir = 'mazes'
    os.makedirs(output_dir, exist_ok=True) # Create output directory
    total_mazes = 0

    print("Starting maze generation...")

    for size in sizes:
        print(f"Generating {size}x{size} mazes...")
        for i in range(10):
            start_time = time.time()
            # Generate the grid directly
            grid = generate_maze(size)
            generation_time = time.time() - start_time

            # Create a Mazes instance
            maze_instance = Mazes(
                size=size,
                maze_number=i + 1,
                grid=grid,
                generation_time=generation_time
            )

            # Save the Mazes instance to a JSON file
            filename = os.path.join(output_dir, f"maze_{size}x{size}_{i+1}.json")
            maze_instance.save(filename)
            total_mazes += 1
            print(f"Generated and saved {filename}")

    # Remove DataFrame and CSV saving logic
    print(f"\nAll mazes have been generated and saved to the '{output_dir}' directory.")
    print(f"Total mazes generated: {total_mazes}")

if __name__ == "__main__":
    main() 