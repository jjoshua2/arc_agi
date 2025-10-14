import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = np.array(grid_lst)
    h, w = grid.shape  # h: height (rows), w: width (cols)
    # Find seeds: non-zero positions
    seeds = [(i, j, grid[i, j]) for i in range(h) for j in range(w) if grid[i, j] != 0]
    if len(seeds) != 2:
        # Assuming always 2 as per examples; copy if not
        return grid.tolist()
    seed1, seed2 = seeds
    result = np.zeros((h, w), dtype=int)
    if w >= h:
        # Vertical stripes: sort by column (j)
        if seed1[1] > seed2[1]:
            seed1, seed2 = seed2, seed1
        pos1, k1 = seed1[1], seed1[2]
        pos2, k2 = seed2[1], seed2[2]
        step = pos2 - pos1
        i = 0
        while True:
            pos = pos1 + i * step
            if pos >= w:
                break
            k = k1 if i % 2 == 0 else k2
            result[:, pos] = k
            i += 1
    else:
        # Horizontal stripes: sort by row (i)
        if seed1[0] > seed2[0]:
            seed1, seed2 = seed2, seed1
        pos1, k1 = seed1[0], seed1[2]
        pos2, k2 = seed2[0], seed2[2]
        step = pos2 - pos1
        i = 0
        while True:
            pos = pos1 + i * step
            if pos >= h:
                break
            k = k1 if i % 2 == 0 else k2
            result[pos, :] = k
            i += 1
    return result.tolist()