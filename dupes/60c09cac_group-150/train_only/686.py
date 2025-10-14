import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    h, w = len(grid_lst), len(grid_lst[0])
    grid = np.array(grid_lst)
    out = np.zeros((2 * h, 2 * w), dtype=int)
    for i in range(h):
        for j in range(w):
            out[2*i:2*i+2, 2*j:2*j+2] = grid[i, j]
    return out.tolist()