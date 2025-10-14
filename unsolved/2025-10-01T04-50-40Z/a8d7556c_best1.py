def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    grid = [row[:] for row in grid_lst]
    rows = len(grid)
    cols = len(grid[0])
    for i in range(rows - 1):
        for j in range(cols - 1):
            if (grid[i][j] == 0 and
                grid[i][j + 1] == 0 and
                grid[i + 1][j] == 0 and
                grid[i + 1][j + 1] == 0):
                grid[i][j] = 2
                grid[i][j + 1] = 2
                grid[i + 1][j] = 2
                grid[i + 1][j + 1] = 2
    return grid