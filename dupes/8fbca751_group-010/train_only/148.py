def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return [row[:] for row in grid]
    
    rows = len(grid)
    cols = len(grid[0])
    
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    new_grid = [row[:] for row in grid]
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 8 and not visited[i][j]:
                # Flood fill to get the 8-connected component
                positions = []
                stack = [(i, j)]
                visited[i][j] = True
                positions.append((i, j))
                
                while stack:
                    x, y = stack.pop()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and grid[nx][ny] == 8:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
                            positions.append((nx, ny))
                
                if positions:
                    min_r = min(x for x, y in positions)
                    max_r = max(x for x, y in positions)
                    min_c = min(y for x, y in positions)
                    max_c = max(y for x, y in positions)
                    
                    # Fill 0s in the bounding box with 2
                    for r in range(min_r, max_r + 1):
                        for c in range(min_c, max_c + 1):
                            if new_grid[r][c] == 0:
                                new_grid[r][c] = 2
    
    return new_grid