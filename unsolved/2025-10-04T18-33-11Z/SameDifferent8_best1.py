import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Find the zero row
    zero_row = None
    for r in range(rows):
        if np.all(grid[r, :] == 0):
            zero_row = r
            break
            
    if zero_row is None:
        return [[0]]
        
    top_region = grid[:zero_row, :]
    bottom_region = grid[zero_row+1:, :]
    
    # Create masks
    top_mask = (top_region != 0).astype(int)
    bottom_mask = (bottom_region != 0).astype(int)
    
    if top_mask.shape != bottom_mask.shape:
        return [[0]]
        
    if not np.array_equal(top_mask, bottom_mask):
        return [[0]]
        
    # Find bounding box of top_mask
    non_zero_indices = np.where(top_mask == 1)
    if len(non_zero_indices[0]) == 0:
        return [[0]]
        
    min_r = np.min(non_zero_indices[0])
    max_r = np.max(non_zero_indices[0])
    min_c = np.min(non_zero_indices[1])
    max_c = np.max(non_zero_indices[1])
    
    output_rows = max_r - min_r + 1
    output_cols = max_c - min_c + 1
    output = np.zeros((output_rows, output_cols), dtype=int)
    
    for r in range(min_r, max_r+1):
        for c in range(min_c, max_c+1):
            if top_mask[r, c] == 1:
                output_r = r - min_r
                output_c = c - min_c
                output[output_r, output_c] = top_region[r, c]
                
    return output.tolist()