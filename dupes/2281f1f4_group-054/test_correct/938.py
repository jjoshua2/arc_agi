def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    h, w = len(grid_lst), len(grid_lst[0])
    grid = [row[:] for row in grid_lst]
    
    # Assume h=10, w=10
    grey_cols = {j for j in range(w) if grid[0][j] == 5}
    grey_rows = {i for i in range(h) if grid[i][w-1] == 5}
    
    for i in grey_rows:
        for j in grey_cols:
            if grid[i][j] == 0:
                grid[i][j] = 2
    
    return grid