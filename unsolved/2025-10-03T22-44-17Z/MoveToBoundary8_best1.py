import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Find all zero positions that are not on the border
    zero_positions = []
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if grid[i, j] == 0:
                zero_positions.append((i, j))
    
    if not zero_positions:
        return grid_lst
    
    # Find the bounding box of the hole
    min_r = min(i for i, j in zero_positions)
    max_r = max(i for i, j in zero_positions)
    min_c = min(j for i, j in zero_positions)
    max_c = max(j for i, j in zero_positions)
    
    hole_height = max_r - min_r + 1
    hole_width = max_c - min_c + 1
    
    # Extract the pattern from the hole's bounding box
    pattern = grid[min_r:max_r+1, min_c:max_c+1].copy()
    
    # Flip the pattern horizontally
    flipped_pattern = np.fliplr(pattern)
    
    # Create the output grid
    output_grid = grid.copy()
    
    # Place the flipped pattern back into the hole
    output_grid[min_r:max_r+1, min_c:max_c+1] = flipped_pattern
    
    return output_grid.tolist()