import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    if rows == 0 or cols == 0:
        return []
    
    center_row = rows / 2.0
    center_col = cols / 2.0
    
    # Find all seed positions (non-zero cells)
    seeds = np.argwhere(grid != 0)
    
    output = grid.copy()
    
    for seed in seeds:
        r, c = seed
        color = grid[r, c]
        
        # Determine vertical direction: up if r < center_row else down
        if r < center_row:
            vert_start = 0
            vert_end = r + 1  # up to and including r
            vert_step = 1
        else:
            vert_start = r
            vert_end = rows
            vert_step = 1
        
        # Fill vertical arm: column c, rows from vert_start to vert_end -1
        for nr in range(vert_start, vert_end):
            output[nr, c] = color
        
        # Determine horizontal direction: left if c < center_col else right
        if c < center_col:
            horiz_start = 0
            horiz_end = c + 1
            horiz_step = 1
        else:
            horiz_start = c
            horiz_end = cols
            horiz_step = 1
        
        # Fill horizontal arm: row r, cols from horiz_start to horiz_end -1
        for nc in range(horiz_start, horiz_end):
            output[r, nc] = color
    
    return output.tolist()