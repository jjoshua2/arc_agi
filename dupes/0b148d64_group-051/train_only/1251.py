def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    
    # Find the highest row that is entirely 0s
    max_all0_row = -1
    for r in range(rows):
        if all(cell == 0 for cell in grid[r]):
            max_all0_row = r
    bottom_start = max_all0_row + 1 if max_all0_row != -1 else 0
    
    if bottom_start >= rows:
        return []
    
    # Visited matrix
    visited = [[False] * cols for _ in range(rows)]
    
    # Directions for 4-connected
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # List of components: (size, min_r, max_r, min_c, max_c)
    components = []
    
    for i in range(bottom_start, rows):
        for j in range(cols):
            if visited[i][j] or grid[i][j] == 0:
                continue
            c = grid[i][j]
            stack = [(i, j)]
            visited[i][j] = True
            comp_size = 1
            min_r, max_r = i, i
            min_c, max_c = j, j
            while stack:
                x, y = stack.pop()
                for dx, dy in directions:
                    nx = x + dx
                    ny = y + dy
                    if bottom_start <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and grid[nx][ny] == c:
                        visited[nx][ny] = True
                        stack.append((nx, ny))
                        comp_size += 1
                        min_r = min(min_r, nx)
                        max_r = max(max_r, nx)
                        min_c = min(min_c, ny)
                        max_c = max(max_c, ny)
            components.append((comp_size, min_r, max_r, min_c, max_c))
    
    if not components:
        # No components, return empty or original bottom? But assume there is
        return []
    
    # Find the component with maximum size (if tie, pick first, assume unique)
    best = max(components, key=lambda x: x[0])
    _, min_r, max_r, min_c, max_c = best
    
    out_h = max_r - min_r + 1
    out_w = max_c - min_c + 1
    
    # Extract the subgrid
    output = []
    for r in range(min_r, max_r + 1):
        row = []
        for c in range(min_c, max_c + 1):
            row.append(grid[r][c])
        output.append(row)
    
    return output