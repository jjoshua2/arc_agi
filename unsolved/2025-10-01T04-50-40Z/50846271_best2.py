def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = [row[:] for row in grid_lst]
    rows = len(grid)
    cols = len(grid[0])
    
    # Horizontal: process each row
    for i in range(rows):
        two_cols = [j for j in range(cols) if grid[i][j] == 2]
        if len(two_cols) >= 2:
            min_j = min(two_cols)
            max_j = max(two_cols)
            # Fill 5's between min_j and max_j
            for j in range(min_j, max_j + 1):
                if grid[i][j] == 5:
                    grid[i][j] = 8
            # Extend right if possible
            if max_j + 1 < cols and grid[i][max_j + 1] == 5:
                grid[i][max_j + 1] = 8
    
    # Vertical: process each column
    for j in range(cols):
        two_rows = [i for i in range(rows) if grid[i][j] == 2]
        if len(two_rows) >= 2:
            min_i = min(two_rows)
            max_i = max(two_rows)
            # Fill 5's between min_i and max_i
            for i in range(min_i, max_i + 1):
                if grid[i][j] == 5:
                    grid[i][j] = 8
            # Extend down if possible
            if max_i + 1 < rows and grid[max_i + 1][j] == 5:
                grid[max_i + 1][j] = 8
    
    return grid