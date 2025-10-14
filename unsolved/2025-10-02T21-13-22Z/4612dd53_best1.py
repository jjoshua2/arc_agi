import sys

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Find all positions with 1
    ones = [(i, j) for i in range(rows) for j in range(cols) if grid[i][j] == 1]
    
    if not ones:
        return [row[:] for row in grid]
    
    min_r = min(pos[0] for pos in ones)
    max_r = max(pos[0] for pos in ones)
    min_c = min(pos[1] for pos in ones)
    max_c = max(pos[1] for pos in ones)
    
    output = [row[:] for row in grid]
    
    # Fill top row
    for c in range(min_c, max_c + 1):
        if output[min_r][c] == 0:
            output[min_r][c] = 2
    
    # Fill bottom row
    for c in range(min_c, max_c + 1):
        if output[max_r][c] == 0:
            output[max_r][c] = 2
    
    # Fill left column
    for r in range(min_r, max_r + 1):
        if output[r][min_c] == 0:
            output[r][min_c] = 2
    
    # Fill right column
    for r in range(min_r, max_r + 1):
        if output[r][max_c] == 0:
            output[r][max_c] = 2
    
    return output