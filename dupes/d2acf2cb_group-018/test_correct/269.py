def toggle(color):
    if color == 0:
        return 8
    elif color == 8:
        return 0
    elif color == 6:
        return 7
    elif color == 7:
        return 6
    else:
        return color

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    grid = [row[:] for row in grid_lst]
    rows = len(grid)
    if rows == 0:
        return grid
    cols = len(grid[0])
    if cols < 2:
        return grid
    
    # Handle horizontal segments
    for r in range(rows):
        if grid[r][0] == 4 and grid[r][cols - 1] == 4:
            for c in range(1, cols - 1):
                grid[r][c] = toggle(grid[r][c])
    
    # Handle vertical segments
    for c in range(cols):
        if grid[0][c] == 4 and grid[rows - 1][c] == 4:
            for r in range(1, rows - 1):
                grid[r][c] = toggle(grid[r][c])
    
    return grid