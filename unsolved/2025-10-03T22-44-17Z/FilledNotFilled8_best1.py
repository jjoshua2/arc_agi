def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Create a copy
    result = [row[:] for row in grid]
    
    for j in range(cols):
        non_zero_rows = [i for i in range(rows) if grid[i][j] != 0]
        if not non_zero_rows:
            continue
        
        # Assume all non-zero have same color c
        c = grid[non_zero_rows[0]][j]
        
        # Verify all same color (for safety, though not necessary for this puzzle)
        if any(grid[i][j] != c for i in non_zero_rows):
            # If not, perhaps skip or handle differently, but assuming uniform
            continue
        
        S = set(non_zero_rows)
        min_r = min(S)
        max_r = max(S)
        span = max_r - min_r + 1
        num = len(S)
        
        if num < span:
            # Fill the gaps
            for i in range(min_r, max_r + 1):
                result[i][j] = c
        elif num == span:
            # Hollow out the inner part
            for i in range(min_r + 1, max_r):
                result[i][j] = 0
            # The boundaries min_r and max_r remain c (already are)
    
    return result