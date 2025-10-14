def transform(input_grid: list[list[int]]) -> list[list[int]]:
    if not input_grid or not input_grid[0]:
        return [row[:] for row in input_grid]
    
    rows = len(input_grid)
    cols = len(input_grid[0])
    grid = [row[:] for row in input_grid]
    
    # Directions for 8-neighbors
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    # Find all positions with 1
    ones_positions = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]
    
    for r, c in ones_positions:
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                grid[nr][nc] = 1
    
    return grid