def transform(grid_lst: List[List[int]]) -> List[List[int]]:
    import numpy as np
    grid = np.array(grid_lst)
    purple = 8
    # Find the bounding box of the purple region
    purple_mask = (grid == purple)
    if not np.any(purple_mask):
        return grid.tolist()
    
    min_row, min_col = np.min(np.argwhere(purple_mask), axis=0)
    max_row, max_col = np.max(np.argwhere(purple_mask), axis=0)
    
    result = grid.copy()
    
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            color = grid[r][c]
            if color != 0 and color != purple:
                # Check if this cell is outside the purple region
                if not (min_row <= r <= max_row and min_col <= c <= max_col):
                    # It is outside. Now determine where to add the new cell.
                    if c >= min_col and c <= max_col:
                        if r < min_row:
                            # Above: add to the top row
                            result[min_row, c] = color
                        elif r > max_row:
                            # Below: add to the bottom row
                            result[max_row, c] = color
                    if r >= min_row and r <= max_row:
                        if c < min_col:
                            # Left: add to the leftmost column
                            result[r, min_col] = color
                        elif c > max_col:
                            # Right: add to the rightmost column
                            result[r, max_col] = color
    
    return result.tolist()