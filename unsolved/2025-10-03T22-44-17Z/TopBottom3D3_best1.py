def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    rows = len(grid)
    cols = len(grid[0])
    
    # Count frequency of non-zero colors
    color_count = {}
    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            if val != 0:
                color_count[val] = color_count.get(val, 0) + 1
    
    if not color_count:
        return [row[:] for row in grid]
    
    # Find the fill color F: the most common non-zero color
    F = max(color_count, key=color_count.get)
    
    # Directions for neighbors
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Create output grid
    output = [row[:] for row in grid]
    
    # For each cell that is not 0 and not F
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and grid[r][c] != F:
                # Count adjacent F cells
                count = 0
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == F:
                        count += 1
                if count >= 2:
                    output[r][c] = F
                else:
                    output[r][c] = 0
    
    return output