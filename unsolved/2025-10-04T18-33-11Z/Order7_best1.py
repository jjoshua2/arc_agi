import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    h, w = grid.shape
    
    # Find the gray column (color 5)
    gray_col = None
    for c in range(w):
        if np.any(grid[:, c] == 5):
            gray_col = c
            break
    
    if gray_col is None:
        return grid.tolist()
    
    # Create output grid
    output = np.zeros_like(grid)
    
    # Copy the gray column
    output[:, gray_col] = grid[:, gray_col]
    
    # Mirror colored blocks (color 4) across the gray column
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 4:
                # Calculate mirror position
                mirror_c = 2 * gray_col - c
                if 0 <= mirror_c < w:
                    output[r, mirror_c] = 4
    
    return output.tolist()