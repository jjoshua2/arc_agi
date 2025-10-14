import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Find all non-zero cells
    non_zero_indices = np.where(grid != 0)
    if len(non_zero_indices[0]) == 0:
        return [[]]
    
    min_row = np.min(non_zero_indices[0])
    max_row = np.max(non_zero_indices[0])
    min_col = np.min(non_zero_indices[1])
    max_col = np.max(non_zero_indices[1])
    
    # Extract the bounding box
    result = grid[min_row:max_row+1, min_col:max_col+1]
    
    return result.tolist()