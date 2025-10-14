import numpy as np
from collections import deque

def transform(grid_lst):
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Find the two most common colors (the background colors)
    flat_grid = grid.flatten()
    unique, counts = np.unique(flat_grid, return_counts=True)
    # Sort by frequency and get the two most common
    sorted_indices = np.argsort(-counts)
    main_colors = unique[sorted_indices[:2]].tolist()
    
    # Create a mask for seeds: cells that are not one of the two main colors
    seed_mask = ~np.isin(grid, main_colors)
    
    # Create output grid as a copy of input
    output = grid.copy()
    
    # For each seed cell, propagate its color along the four diagonals
    # within cells of the same checkerboard color as the seed's original position
    for r in range(rows):
        for c in range(cols):
            if seed_mask[r, c]:
                seed_color = grid[r, c]
                # The seed's original cell color determines which checkerboard pattern to follow
                # We'll propagate along diagonals within cells of the same color as the seed's original cell
                target_color = grid[r, c]
                
                # Directions: four diagonals
                directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < rows and 0 <= nc < cols:
                        if grid[nr, nc] == target_color:
                            output[nr, nc] = seed_color
                        else:
                            break  # Stop when we hit a different color
                        nr += dr
                        nc += dc
    
    return output.tolist()