import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    n = len(grid_lst)
    c = grid_lst[0][0]
    output = np.zeros((15, 15), dtype=int)
    step = n + 1
    for i in range(15):
        for j in range(15):
            if (i % step == n) or (j % step == n):
                output[i, j] = c
    return output.tolist()