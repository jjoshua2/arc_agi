import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    
    # Find rows and columns with non-zero values
    non_zero_mask = grid != 0
    rows_with_non_zero = np.any(non_zero_mask, axis=1)
    cols_with_non_zero = np.any(non_zero_mask, axis=0)
    
    # Find the bounding box
    if np.any(rows_with_non_zero) and np.any(cols_with_non_zero):
        min_row = np.where(rows_with_non_zero)[0][0]
        max_row = np.where(rows_with_non_zero)[0][-1]
        min_col = np.where(cols_with_non_zero)[0][0]
        max_col = np.where(cols_with_non_zero)[0][-1]
        
        # Extract the bounding box
        result = grid[min_row:max_row+1, min_col:max_col+1]
    else:
        # If no non-zero values, return empty grid
        result = np.array([[]])
    
    return result.tolist()