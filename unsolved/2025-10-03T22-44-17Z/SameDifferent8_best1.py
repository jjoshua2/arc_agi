import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return [[0]]  # edge case, but unlikely
    
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    min_score = float('inf')
    best_c = 0
    best_i = -1
    best_j = -1
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if grid[i, j] != 4:
                continue
            up = grid[i - 1, j]
            down = grid[i + 1, j]
            left = grid[i, j - 1]
            right = grid[i, j + 1]
            if up == down == left == right and up != 4:
                c = up
                score = i + j
                if score < min_score or \
                   (score == min_score and (i < best_i or (i == best_i and j < best_j))):
                    min_score = score
                    best_c = c
                    best_i = i
                    best_j = j
    
    if best_c == 0:
        # If none found, perhaps error, but assume always one
        return [[0]]
    
    return [[best_c]]