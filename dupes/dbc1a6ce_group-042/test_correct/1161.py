def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    
    # Horizontal fills: for each row, fill gaps between consecutive 1s
    for r in range(rows):
        one_cols = [c for c in range(cols) if grid[r][c] == 1]
        if len(one_cols) < 2:
            continue
        one_cols.sort()
        for i in range(len(one_cols) - 1):
            start_c = one_cols[i] + 1
            end_c = one_cols[i + 1] - 1
            for c in range(start_c, end_c + 1):
                output[r][c] = 8
    
    # Vertical fills: for each column, fill gaps between consecutive 1s
    for c in range(cols):
        one_rows = [r for r in range(rows) if grid[r][c] == 1]
        if len(one_rows) < 2:
            continue
        one_rows.sort()
        for i in range(len(one_rows) - 1):
            start_r = one_rows[i] + 1
            end_r = one_rows[i + 1] - 1
            for r in range(start_r, end_r + 1):
                output[r][c] = 8
    
    return output