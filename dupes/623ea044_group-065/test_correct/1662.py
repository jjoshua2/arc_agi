import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Find the seed (non-zero cell)
    seed_pos = None
    seed_color = None
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0:
                seed_pos = (r, c)
                seed_color = grid[r, c]
                break
        if seed_pos is not None:
            break
    
    # If no seed found, return original grid
    if seed_pos is None:
        return grid_lst
    
    seed_r, seed_c = seed_pos
    
    # Create output grid
    output = np.zeros_like(grid)
    
    # Fill diagonals
    main_diag_const = seed_r - seed_c
    anti_diag_const = seed_r + seed_c
    
    for r in range(rows):
        for c in range(cols):
            if (r - c == main_diag_const) or (r + c == anti_diag_const):
                output[r, c] = seed_color
    
    return output.tolist()