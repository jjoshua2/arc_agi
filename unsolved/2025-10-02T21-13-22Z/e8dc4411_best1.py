import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    output = grid.copy()
    
    # For color 1 (blue)
    pos1 = np.argwhere(grid == 1)
    if len(pos1) > 0:
        min_row1 = np.min(pos1[:, 0])
        min_col1 = np.min(pos1[:, 1])
        r, c = min_row1 - 1, min_col1 - 1
        while r >= 0 and c >= 0:
            output[r, c] = 1
            r -= 1
            c -= 1
    
    # For color 2 (red)
    pos2 = np.argwhere(grid == 2)
    if len(pos2) > 0:
        max_row2 = np.max(pos2[:, 0])
        max_col2 = np.max(pos2[:, 1])
        r, c = max_row2 + 1, max_col2 + 1
        while r < rows and c < cols:
            output[r, c] = 2
            r += 1
            c += 1
    
    return output.tolist()