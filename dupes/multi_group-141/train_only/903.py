import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    size = 5
    max_zeros = -1
    best_i, best_j = rows, cols  # Initialize to large values for min comparison
    for i in range(rows - size + 1):
        for j in range(cols - size + 1):
            sub = grid[i:i+size, j:j+size]
            num_zeros = np.sum(sub == 0)
            if (num_zeros > max_zeros) or \
               (num_zeros == max_zeros and (i < best_i or (i == best_i and j < best_j))):
                max_zeros = num_zeros
                best_i = i
                best_j = j
    if max_zeros == -1:
        return grid.tolist()
    # Set border to 5
    # Top row
    grid[best_i, best_j:best_j + size] = 5
    # Bottom row
    grid[best_i + size - 1, best_j:best_j + size] = 5
    # Left column, excluding corners
    grid[best_i + 1:best_i + size - 1, best_j] = 5
    # Right column, excluding corners
    grid[best_i + 1:best_i + size - 1, best_j + size - 1] = 5
    return grid.tolist()