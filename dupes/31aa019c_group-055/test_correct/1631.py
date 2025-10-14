import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Create output grid filled with zeros
    output = np.zeros_like(grid)
    
    # Count occurrences of each value
    value_counts = {}
    for i in range(rows):
        for j in range(cols):
            val = grid[i, j]
            value_counts[val] = value_counts.get(val, 0) + 1
    
    # Find unique values (appearing exactly once)
    unique_values = [val for val, count in value_counts.items() if count == 1 and val != 0]
    
    if not unique_values:
        # No unique non-zero values found, return zeros
        return output.tolist()
    
    # Find the smallest unique value
    center_value = min(unique_values)
    
    # Find the position of the center cell
    center_pos = None
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == center_value:
                center_pos = (i, j)
                break
        if center_pos:
            break
    
    if not center_pos:
        return output.tolist()
    
    center_i, center_j = center_pos
    
    # Create 3x3 block around center
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            ni = center_i + di
            nj = center_j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                if di == 0 and dj == 0:
                    # Center keeps original value
                    output[ni, nj] = grid[ni, nj]
                else:
                    # Surrounding cells become 2
                    output[ni, nj] = 2
    
    return output.tolist()