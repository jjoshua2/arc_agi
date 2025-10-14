import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    for r in range(rows - 6):
        for c in range(cols - 6):
            sub = grid[r:r+7, c:c+7]
            C = sub[0, 0]
            if (np.all(sub[0, :] == C) and
                np.all(sub[6, :] == C) and
                np.all(sub[:, 0] == C) and
                np.all(sub[:, 6] == C)):
                return sub.tolist()
    # If no such subgrid found, return empty or original, but assuming always one
    return []