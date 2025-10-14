import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Create a copy of the grid to modify
    result = grid.copy()
    
    # Find all colored cells (non-zero values)
    colored_cells = []
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0:
                colored_cells.append((r, c, grid[r, c]))
    
    # For each colored cell, create mirror images
    for r, c, color in colored_cells:
        # Original position already has the color
        # Create vertical mirror
        result[r, cols-1-c] = color
        # Create horizontal mirror
        result[rows-1-r, c] = color
        # Create both axes mirror
        result[rows-1-r, cols-1-c] = color
    
    return result.tolist()