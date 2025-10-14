import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    output = np.copy(grid)
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 9:
                output[i][j] = grid[j][i]
    return output.tolist()