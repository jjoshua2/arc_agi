def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or len(grid) != 2:
        return grid
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Create a copy of the grid
    result = [row[:] for row in grid]
    
    # Apply the alternating pattern
    for j in range(cols):
        if j % 2 == 1:  # odd columns get swapped
            result[0][j] = grid[1][j]
            result[1][j] = grid[0][j]
    
    return result