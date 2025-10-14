def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Check for full uniform non-zero columns
    for c in range(cols):
        col_color = grid[0][c]
        if col_color == 0:
            continue
        is_uniform = all(grid[r][c] == col_color for r in range(rows))
        if is_uniform:
            return [[col_color]]
    
    # Check for full uniform non-zero rows
    for r in range(rows):
        row_color = grid[r][0]
        if row_color == 0:
            continue
        is_uniform = all(grid[r][c] == row_color for c in range(cols))
        if is_uniform:
            return [[row_color]]
    
    # If no full uniform non-zero row or column found, return empty or default, but assuming always one
    return [[0]]  # fallback, but shouldn't happen