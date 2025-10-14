def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    grid = [row[:] for row in grid_lst]
    
    non_zeros = [(i, j) for i in range(rows) for j in range(cols) if grid[i][j] != 0]
    if not non_zeros:
        return grid
    
    min_r = min(p[0] for p in non_zeros)
    max_r = max(p[0] for p in non_zeros)
    min_c = min(p[1] for p in non_zeros)
    max_c = max(p[1] for p in non_zeros)
    
    # Find gap rows
    gap_rows = []
    for r in range(min_r, max_r + 1):
        if all(grid[r][c] == 0 for c in range(min_c, max_c + 1)):
            gap_rows.append(r)
    
    # Find gap columns
    gap_cols = []
    for c in range(min_c, max_c + 1):
        if all(grid[r][c] == 0 for r in range(min_r, max_r + 1)):
            gap_cols.append(c)
    
    # Fill gap rows entirely with 3
    for r in gap_rows:
        for j in range(cols):
            grid[r][j] = 3
    
    # Fill gap columns entirely with 3
    for c in gap_cols:
        for i in range(rows):
            grid[i][c] = 3
    
    return grid