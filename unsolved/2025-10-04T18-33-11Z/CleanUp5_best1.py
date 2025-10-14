import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Find the region with highest density of non-zero values
    # Calculate density for each row
    row_density = np.sum(grid != 0, axis=1)
    # Calculate density for each column
    col_density = np.sum(grid != 0, axis=0)
    
    # Find rows with highest density (top 50%)
    row_threshold = np.percentile(row_density, 50)
    high_density_rows = np.where(row_density >= row_threshold)[0]
    
    # Find columns with highest density (top 50%)
    col_threshold = np.percentile(col_density, 50)
    high_density_cols = np.where(col_density >= col_threshold)[0]
    
    # Create mask for main pattern region
    pattern_mask = np.zeros((rows, cols), dtype=bool)
    for r in high_density_rows:
        for c in high_density_cols:
            pattern_mask[r, c] = True
    
    # Find the most common color in the pattern region
    pattern_values = grid[pattern_mask]
    non_zero_pattern = pattern_values[pattern_values != 0]
    if len(non_zero_pattern) > 0:
        most_common_color = np.bincount(non_zero_pattern).argmax()
    else:
        most_common_color = 0
    
    # Create output grid
    output = np.zeros_like(grid)
    
    # Copy pattern region with most common color
    for r in range(rows):
        for c in range(cols):
            if pattern_mask[r, c] and grid[r, c] != 0:
                output[r, c] = most_common_color
            elif pattern_mask[r, c] and grid[r, c] == 0:
                output[r, c] = 0
            else:
                output[r, c] = 0
    
    return output.tolist()