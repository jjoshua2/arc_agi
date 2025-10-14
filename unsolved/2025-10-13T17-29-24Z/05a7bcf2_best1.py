def transform(grid):
    if not grid or not grid[0]:
        return grid
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]  # Make a copy

    # Find the min and max row and column that have non-zero values
    min_r, max_r = rows, -1
    min_c, max_c = cols, -1
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

    # For rows between min_r and max_r (exclusive), if the row is entirely zero, set each cell to twice the value from the min_r row
    for r in range(rows):
        if min_r < r < max_r:
            if all(grid[r][c] == 0 for c in range(cols)):
                for c in range(cols):
                    output[r][c] = 2 * grid[min_r][c] 

    # For columns between min_c and max_c (exclusive), if the column is entirely zero, set each cell to twice the value from the min_c column
    for c in range(cols):
        if min_c < c < max_c:
            if all(grid[r][c] == 0 for r in range(rows)):
                for r in range(rows):
                    output[r][c] = 2 * grid[r][min_c] 
    
    return output