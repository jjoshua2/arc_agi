def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    import numpy as np
    grid = np.array(grid_lst)
    purple_color = 8
    
    # Find the bounding box of the purple region
    purple_mask = (grid == purple_color)
    if not np.any(purple_mask):
        return grid.tolist()
    
    # Get the min and max coordinates of the purple region
    rows, cols = np.where(purple_mask)
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    
    # Create a copy of the grid to modify
    result = grid.copy()
    
    # For each cell that is not purple and not zero, project it to the boundary of the purple region
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            color = grid[r, c]
            if color != 0 and color != purple_color:
                # Check if this cell is above, below, left, or right of the purple region
                if c >= min_col and c <= max_col:
                    if r < min_row:
                        # Above: project to the top boundary
                        result[min_row, c] = color
                    elif r > max_row:
                        # Below: project to the bottom boundary
                        result[max_row, c] = color
                if r >= min_row and r <= max_row:
                    if c < min_col:
                        # Left: project to the left boundary
                        result[r, min_col] = color
                    elif c > max_col:
                        # Right: project to the right boundary
                        result[r, max_col] = color
    
    return result.tolist()