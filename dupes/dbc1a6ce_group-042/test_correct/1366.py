def transform(grid):
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]
    
    # Horizontal fills: for each row
    for r in range(rows):
        blue_cols = [c for c in range(cols) if grid[r][c] == 1]
        if len(blue_cols) < 2:
            continue
        blue_cols.sort()
        for i in range(len(blue_cols) - 1):
            for c in range(blue_cols[i] + 1, blue_cols[i + 1]):
                result[r][c] = 8
    
    # Vertical fills: for each column
    for c in range(cols):
        blue_rows = [r for r in range(rows) if grid[r][c] == 1]
        if len(blue_rows) < 2:
            continue
        blue_rows.sort()
        for i in range(len(blue_rows) - 1):
            for r in range(blue_rows[i] + 1, blue_rows[i + 1]):
                result[r][c] = 8
    
    return result