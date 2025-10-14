import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Create output grid initialized to zeros
    output = np.zeros_like(grid)
    
    # Find all unique colors (excluding 0)
    colors = np.unique(grid)
    colors = colors[colors != 0]
    
    # For each color, find top-leftmost occurrence and create 2x2 block
    for color in colors:
        # Find all positions with this color
        positions = np.argwhere(grid == color)
        
        if len(positions) == 0:
            continue
            
        # Find top-leftmost position (min row, then min col)
        top_left = min(positions, key=lambda x: (x[0], x[1]))
        r, c = top_left
        
        # Create 2x2 block if possible
        if r + 1 < rows and c + 1 < cols:
            output[r, c] = color
            output[r, c + 1] = color
            output[r + 1, c] = color
            output[r + 1, c + 1] = color
    
    return output.tolist()