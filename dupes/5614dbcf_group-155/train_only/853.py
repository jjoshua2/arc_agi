import numpy as np
from collections import Counter

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = np.array(grid_lst)
    h, w = grid.shape
    
    # Find vertical boundaries (between columns)
    col_boundaries = []
    for c in range(w - 1):
        changes = np.mean(grid[:, c] != grid[:, c + 1])
        if changes > 0.5:
            col_boundaries.append(c)
    col_starts = [0] + [c + 1 for c in col_boundaries]
    col_ends = col_starts[1:] + [w]
    
    # Find horizontal boundaries (between rows)
    row_boundaries = []
    for r in range(h - 1):
        changes = np.mean(grid[r, :] != grid[r + 1, :])
        if changes > 0.5:
            row_boundaries.append(r)
    row_starts = [0] + [r + 1 for r in row_boundaries]
    row_ends = row_starts[1:] + [h]
    
    # Number of row and column blocks
    m = len(row_starts)
    k = len(col_starts)
    
    # Build output
    output = [[0] * k for _ in range(m)]
    for i in range(m):
        rs, re = row_starts[i], row_ends[i]
        for j in range(k):
            cs, ce = col_starts[j], col_ends[j]
            sub = grid[rs:re, cs:ce]
            flat = sub.flatten()
            if len(flat) == 0:
                mode = 0
            else:
                mode = Counter(flat).most_common(1)[0][0]
            output[i][j] = mode
    
    return output