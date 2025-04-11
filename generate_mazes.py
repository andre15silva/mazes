import numpy as np
import pandas as pd
from mazelib import Maze
from mazelib.generate.Prims import Prims
import time

def generate_maze(size):
    """Generate a maze of given size using Prim's algorithm."""
    m = Maze()
    m.generator = Prims(size, size)
    m.generate()
    return m.grid

def maze_to_string(maze):
    """Convert maze numpy array to string representation."""
    # Convert to flat string of 0s and 1s
    return ''.join(str(int(x)) for x in maze.flatten())

def main():
    sizes = [5, 10, 25, 50, 100]
    mazes_data = []
    
    print("Starting maze generation...")
    
    for size in sizes:
        print(f"Generating {size}x{size} mazes...")
        for i in range(10):
            start_time = time.time()
            maze = generate_maze(size)
            generation_time = time.time() - start_time
            
            maze_data = {
                'size': size,
                'maze_number': i + 1,
                'maze': maze_to_string(maze),
                'generation_time': generation_time
            }
            mazes_data.append(maze_data)
            print(f"Generated maze {i + 1}/10 for size {size}x{size}")
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(mazes_data)
    output_file = 'generated_mazes.csv'
    df.to_csv(output_file, index=False)
    print(f"\nAll mazes have been generated and saved to {output_file}")
    print(f"Total mazes generated: {len(mazes_data)}")

if __name__ == "__main__":
    main() 