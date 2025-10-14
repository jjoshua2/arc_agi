def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = [row[:] for row in grid_lst]
    rows = len(grid)
    cols = len(grid[0])
    if rows < 2 or cols < 2:
        return grid
    c = grid[1][1]
    if c == 0:
        return grid
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == c and not (i == 1 and j == 1):
                grid[i][j] = 0
    return grid