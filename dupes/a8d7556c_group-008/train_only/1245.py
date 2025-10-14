def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    for i in range(rows - 1):
        for j in range(cols - 1):
            if (grid[i][j] == 0 and
                grid[i][j + 1] == 0 and
                grid[i + 1][j] == 0 and
                grid[i + 1][j + 1] == 0):
                output[i][j] = 2
                output[i][j + 1] = 2
                output[i + 1][j] = 2
                output[i + 1][j + 1] = 2
    return output