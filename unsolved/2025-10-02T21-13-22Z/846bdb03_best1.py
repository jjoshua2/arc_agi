import numpy as np

def transform(grid_lst):
    grid = np.array(grid_lst)
    
    # Find all non-zero colors and their frequencies
    non_zero_colors = grid[grid != 0]
    if len(non_zero_colors) == 0:
        return [[]]
    
    unique, counts = np.unique(non_zero_colors, return_counts=True)
    
    # Select the color with minimum frequency (if tie, smallest color value)
    min_count = np.min(counts)
    candidates = unique[counts == min_count]
    target_color = np.min(candidates)  # If multiple colors have same min count, pick smallest
    
    # Find positions of target color
    positions = np.argwhere(grid == target_color)
    
    if len(positions) == 0:
        return [[]]
    
    # Get bounding box
    min_row, min_col = np.min(positions, axis=0)
    max_row, max_col = np.max(positions, axis=0)
    
    # Extract the pattern within the bounding box
    result = grid[min_row:max_row+1, min_col:max_col+1].copy()
    
    # Set all non-target colors to 0
    result[result != target_color] = 0
    
    return result.tolist()