import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    purple_color = 8

    # Find the bounding box of the purple rectangle
    purple_shape = np.argwhere(grid == purple_color)
    if len(purple_shape) == 0:
        return grid.tolist()
    min_row, min_col = np.min(purple_shape, axis=0)
    max_row, max_col = np.max(purple_shape, axis=0)

    # Iterate over the grid to find colored cells around the purple rectangle
    rows, cols = grid.shape
    for row in range(rows):
        for col in range(cols):
            color = grid[row, col]
            if color != 0 and color != purple_color:
                # If the colored cell is above/below the central shape
                if min_col <= col <= max_col:
                    if row < min_row:
                        grid[min_row, col] = color
                    elif row > max_row:
                        grid[max_row, col] = color
                # If the colored cell is to the left/right of the central shape
                if min_row <= row <= max_row:
                    if col < min_col:
                        grid[row, min_col] = color
                    elif col > max_col:
                        grid[row, max_col] = color

    return grid.tolist()