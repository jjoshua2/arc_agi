def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid:
        return []
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]
    # Horizontal gap filling
    for r in range(rows):
        two_cols = [c for c in range(cols) if grid[r][c] == 2]
        if two_cols:
            min_c = min(two_cols)
            max_c = max(two_cols)
            for c in range(min_c + 1, max_c):
                if out[r][c] == 5:
                    out[r][c] = 8
    # Vertical gap filling
    for c in range(cols):
        two_rows = [r for r in range(rows) if grid[r][c] == 2]
        if two_rows:
            min_r = min(two_rows)
            max_r = max(two_rows)
            for r in range(min_r + 1, max_r):
                if out[r][c] == 5:
                    out[r][c] = 8
    return out