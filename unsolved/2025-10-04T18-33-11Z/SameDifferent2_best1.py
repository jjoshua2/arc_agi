import numpy as np
from collections import Counter

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    
    # Flatten the grid to count frequencies
    flat_grid = grid.flatten()
    
    # Count frequencies of all colors except 0
    non_zero_counts = Counter(color for color in flat_grid if color != 0)
    
    if not non_zero_counts:
        # If there are no non-zero colors, return all zeros
        return [[0] * len(grid[0]) for _ in range(len(grid))]
    
    # Find the most frequent non-zero color
    frame_color = max(non_zero_counts.items(), key=lambda x: x[1])[0]
    
    # Create output grid
    output = []
    for row in grid:
        new_row = []
        for cell in row:
            if cell == frame_color:
                new_row.append(0)
            else:
                new_row.append(cell)
        output.append(new_row)
    
    return output