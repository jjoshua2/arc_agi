def transform(grid):
    if not grid:
        return grid
    R = len(grid)
    C = len(grid[0])
    k = min(R, C) // 2
    new_grid = []
    for i in range(R):
        new_row = []
        for j in range(C):
            if i + j < k:
                new_row.append(grid[i][j])
            else:
                new_row.append(0)
        new_grid.append(new_row)
    return new_grid