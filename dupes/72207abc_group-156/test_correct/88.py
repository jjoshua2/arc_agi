def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or len(grid) != 3:
        return grid
    row = 1  # The middle row
    width = len(grid[0])
    if width == 0:
        return grid
    
    # Compute triangular positions
    pos = []
    k = 0
    while True:
        t = k * (k + 1) // 2
        if t >= width:
            break
        pos.append(t)
        k += 1
    
    if not pos:
        return grid
    
    # Find the length of the prefix where input is non-zero at triangular positions
    m = 0
    while m < len(pos) and grid[row][pos[m]] != 0:
        m += 1
    
    if m == 0:
        return grid  # No changes if no initial non-zero
    
    # Extract prefix colors
    prefix = [grid[row][pos[j]] for j in range(m)]
    
    # Create a copy of the grid
    new_grid = [row[:] for row in grid]
    
    # Assign colors to triangular positions
    for i in range(len(pos)):
        color_idx = i % m
        new_grid[row][pos[i]] = prefix[color_idx]
    
    return new_grid