def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    
    # Compute min_r, max_r, min_c, max_c
    min_r = rows
    max_r = -1
    min_c = cols
    max_c = -1
    has_nonzero = False
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                has_nonzero = True
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    
    if not has_nonzero:
        return output
    
    # Fill rows
    for i in range(rows):
        if min_r < i < max_r:
            if all(grid[i][j] == 0 for j in range(cols)):
                for j in range(cols):
                    output[i][j] = 3
    
    # Fill columns
    for j in range(cols):
        if min_c < j < max_c:
            if all(grid[r][j] == 0 for r in range(rows)):
                for r in range(rows):
                    output[r][j] = 3
    
    return output