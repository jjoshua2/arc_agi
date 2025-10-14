def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])
    # Find unique columns with at least one 0
    cols_with_zero = set()
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0:
                cols_with_zero.add(c)
    sorted_cols = sorted(cols_with_zero)
    # Assign colors 1 to len(sorted_cols)
    col_to_color = {col: i+1 for i, col in enumerate(sorted_cols)}
    # Create output
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if out[r][c] == 0:
                out[r][c] = col_to_color[c]
    return out