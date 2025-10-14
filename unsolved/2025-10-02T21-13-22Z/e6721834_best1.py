import numpy as np
from collections import Counter

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    
    # Find all non-zero cells and their colors
    non_zero_mask = grid != 0
    non_zero_values = grid[non_zero_mask]
    
    if len(non_zero_values) == 0:
        return [[0]]  # Return single 0 if no non-zero cells
    
    # Find the most common non-zero color
    color_counts = Counter(non_zero_values)
    dominant_color = color_counts.most_common(1)[0][0]
    
    # Find positions of cells with dominant color
    dominant_positions = np.where(grid == dominant_color)
    rows, cols = dominant_positions
    
    if len(rows) == 0:
        return [[0]]  # Return single 0 if no cells with dominant color
    
    # Find bounding box of dominant color cells
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    
    # Create output grid with the bounding box dimensions
    output_rows = max_row - min_row + 1
    output_cols = max_col - min_col + 1
    output_grid = np.zeros((output_rows, output_cols), dtype=int)
    
    # Place dominant color cells in their relative positions
    for r, c in zip(rows, cols):
        output_grid[r - min_row, c - min_col] = dominant_color
    
    return output_grid.tolist()