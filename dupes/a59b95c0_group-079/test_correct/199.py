import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    colors = set(grid.flatten())
    K = len(colors)
    n = grid.shape[0]
    out = np.tile(grid, (K, K))
    return out.tolist()