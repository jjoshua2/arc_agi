import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    
    # Find all positions with non-zero values
    non_zero_positions = np.argwhere(grid != 0)
    
    if len(non_zero_positions) == 0:
        return [[]]
    
    # Get the bounding box of non-zero values
    min_row = np.min(non_zero_positions[:, 0])
    max_row = np.max(non_zero_positions[:, 0])
    min_col = np.min(non_zero_positions[:, 1])
    max_col = np.max(non_zero_positions[:, 1])
    
    # Extract the subgrid
    result = grid[min_row:max_row+1, min_col:max_col+1]
    
    return result.tolist()