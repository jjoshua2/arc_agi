import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Find the bottom-most non-zero row
    bottom_row = None
    for r in range(rows-1, -1, -1):
        if np.any(grid[r] != 0):
            bottom_row = r
            break
    
    if bottom_row is None:
        return []  # No non-zero cells
    
    # Find the top of the small grid (go upwards until we hit a zero row)
    top_row = bottom_row
    for r in range(bottom_row, -1, -1):
        if np.any(grid[r] != 0):
            top_row = r
        else:
            break
    
    # Now, for the rows between top_row and bottom_row, find the left and right boundaries
    left = cols
    right = -1
    for r in range(top_row, bottom_row+1):
        non_zero_indices = np.where(grid[r] != 0)[0]
        if len(non_zero_indices) > 0:
            left = min(left, non_zero_indices[0])
            right = max(right, non_zero_indices[-1])
    
    # Extract the small grid
    small_grid = grid[top_row:bottom_row+1, left:right+1]
    h, w = small_grid.shape
    
    # Create output with checkerboard pattern
    output = np.zeros_like(small_grid)
    for i in range(h):
        for j in range(w):
            if (i + j) % 2 == 0:
                output[i, j] = small_grid[i, j]
            else:
                output[i, j] = 0
    
    return output.tolist()