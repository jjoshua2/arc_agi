def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Copy the grid
    output = [row[:] for row in grid]
    
    # Find non-zero colors
    non_zeros = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                non_zeros.add(grid[r][c])
    
    if len(non_zeros) != 2:
        return output  # No change if not exactly two non-zero colors
    
    low, high = min(non_zeros), max(non_zeros)
    
    # For each column
    for c in range(cols):
        # Find seeds (rows with high)
        seeds = [r for r in range(rows) if grid[r][c] == high]
        for s in seeds:
            # Propagate upwards from s-1 to 0
            for r in range(s - 1, -1, -1):
                if grid[r][c] == 0:
                    break
                if grid[r][c] == low:
                    output[r][c] = high
                else:
                    break  # Stop if high or other non-low, non-0
    
    return output