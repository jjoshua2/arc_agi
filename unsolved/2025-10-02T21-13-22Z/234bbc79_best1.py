import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    h, w = grid.shape
    
    # Remove columns that are all zeros
    non_zero_cols = []
    for c in range(w):
        if not np.all(grid[:, c] == 0):
            non_zero_cols.append(c)
    grid = grid[:, non_zero_cols]
    h, w = grid.shape
    
    # Replace 5s with nearest non-zero neighbor
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 5:
                # Check left
                if c > 0 and grid[r, c-1] != 0 and grid[r, c-1] != 5:
                    grid[r, c] = grid[r, c-1]
                    continue
                # Check right
                if c < w-1 and grid[r, c+1] != 0 and grid[r, c+1] != 5:
                    grid[r, c] = grid[r, c+1]
                    continue
                # Check below
                if r < h-1 and grid[r+1, c] != 0 and grid[r+1, c] != 5:
                    grid[r, c] = grid[r+1, c]
                    continue
                # Check above
                if r > 0 and grid[r-1, c] != 0 and grid[r-1, c] != 5:
                    grid[r, c] = grid[r-1, c]
                    continue
                # If all neighbors are 0 or 5, set to 0
                grid[r, c] = 0
                
    return grid.tolist()