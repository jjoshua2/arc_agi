import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    purple_color = 8
    rows, cols = grid.shape
    
    # Find the bounding box of the purple region
    purple_mask = (grid == purple_color)
    if not np.any(purple_mask):
        return grid_lst
    
    min_row, min_col = np.min(np.argwhere(purple_mask), axis=0)
    max_row, max_col = np.max(np.argwhere(purple_mask), axis=0)
    
    result = grid.copy()
    
    for r in range(rows):
        for c in range(cols):
            if not (min_row <= r <= max_row and min_col <= c <= max_col):  # If outside purple region
                color = grid[r, c]
                if color != 0:
                    if min_col <= c <= max_col:
                        if r < min_row:
                            result[min_row, c] = color
                        elif r > max_row:
                            result[max_row, c] = color
                    if min_row <= r <= max_row:
                        if c < min_col:
                            result[r, min_col] = color
                        elif c > max_col:
                            result[r, max_col] = color
    
    return result.tolist()