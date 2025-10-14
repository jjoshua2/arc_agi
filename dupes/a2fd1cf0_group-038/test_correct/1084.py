def transform(input_grid: list[list[int]]) -> list[list[int]]:
    if not input_grid or not input_grid[0]:
        return [row[:] for row in input_grid]
    
    rows = len(input_grid)
    cols = len(input_grid[0])
    grid = [row[:] for row in input_grid]
    
    # Find position of red (2)
    r_r, c_r = -1, -1
    # Find position of green (3)
    r_g, c_g = -1, -1
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                r_r, c_r = r, c
            elif grid[r][c] == 3:
                r_g, c_g = r, c
    
    if r_r == -1 or r_g == -1:
        return grid  # No markers, return as is
    
    # Horizontal in row r_r, columns from min(c_r, c_g) to max(c_r, c_g)
    min_c = min(c_r, c_g)
    max_c = max(c_r, c_g)
    for c in range(min_c, max_c + 1):
        if c != c_r:
            grid[r_r][c] = 8
    
    # Vertical in column c_g, rows from min(r_r, r_g) to max(r_r, r_g)
    min_r = min(r_r, r_g)
    max_r = max(r_r, r_g)
    for r in range(min_r, max_r + 1):
        if r != r_g:
            grid[r][c_g] = 8
    
    return grid