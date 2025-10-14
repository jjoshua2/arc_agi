import numpy as np
from collections import Counter

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    
    # Count frequency of all non-zero colors
    color_counts = Counter()
    for row in grid:
        for val in row:
            if val != 0:
                color_counts[val] += 1
    
    if not color_counts:
        return grid_lst
    
    # Find the most common non-zero color
    most_common_color = max(color_counts.items(), key=lambda x: x[1])[0]
    
    # Create output grid - remove all cells with the most common color, keep others
    output = np.zeros_like(grid)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] != 0 and grid[i, j] != most_common_color:
                output[i, j] = grid[i, j]
    
    return output.tolist()