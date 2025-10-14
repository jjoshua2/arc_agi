def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])
    output = [row[:] for row in grid]
    for r in range(h - 1):
        for c in range(w - 1):
            if (grid[r][c] == 0 and
                grid[r][c + 1] == 0 and
                grid[r + 1][c] == 0 and
                grid[r + 1][c + 1] == 0):
                output[r][c] = 2
                output[r][c + 1] = 2
                output[r + 1][c] = 2
                output[r + 1][c + 1] = 2
    return output