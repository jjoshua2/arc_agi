def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    h = len(grid)
    w = len(grid[0])
    new_grid = [row[:] for row in grid]
    
    # Find pattern from row 0: columns with 5
    pattern_cols = set()
    for c in range(w):
        if grid[0][c] == 5:
            pattern_cols.add(c)
    
    # For each row i >= 1 (but row 0 unchanged)
    for i in range(1, h):
        if grid[i][w-1] == 5:  # has 5 in last column
            for c in pattern_cols:
                if new_grid[i][c] == 0:
                    new_grid[i][c] = 2
    
    return new_grid