def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Find all positions of 8 (purple)
    purple_positions = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 8]
    
    if not purple_positions:
        return [row[:] for row in grid]
    
    min_r = min(pos[0] for pos in purple_positions)
    max_r = max(pos[0] for pos in purple_positions)
    min_c = min(pos[1] for pos in purple_positions)
    max_c = max(pos[1] for pos in purple_positions)
    
    # Create output as copy
    out = [row[:] for row in grid]
    
    # Fill top
    for c in range(min_c, max_c + 1):
        if out[min_r][c] != 8:
            out[min_r][c] = 1
    
    # Fill bottom
    for c in range(min_c, max_c + 1):
        if out[max_r][c] != 8:
            out[max_r][c] = 1
    
    # Fill left (excluding corners)
    for r in range(min_r + 1, max_r):
        if out[r][min_c] != 8:
            out[r][min_c] = 1
    
    # Fill right (excluding corners)
    for r in range(min_r + 1, max_r):
        if out[r][max_c] != 8:
            out[r][max_c] = 1
    
    return out