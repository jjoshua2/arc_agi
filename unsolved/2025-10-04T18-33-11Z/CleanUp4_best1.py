from typing import List

def transform(grid: List[List[int]]) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0])
    # Make a deep copy of the grid
    result = [row[:] for row in grid]
    
    # First, set all cells with value 8 to 0.
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 8:
                result[i][j] = 0
    
    # Now, for each cell that was originally 8 (now set to 0 in result), 
    # set its four diagonal neighbors to 5.
    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 8:
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < rows and 0 <= nj < cols:
                        result[ni][nj] = 5
    return result