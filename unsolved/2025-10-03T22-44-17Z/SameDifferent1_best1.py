def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 0 and not visited[i][j]:
                component = []
                stack = [(i, j)]
                visited[i][j] = True
                color = grid[i][j]
                while stack:
                    x, y = stack.pop()
                    component.append((x, y))
                    for dx, dy in dirs:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and grid[nx][ny] == color:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
                if component:
                    min_r = min(p[0] for p in component)
                    max_r = max(p[0] for p in component)
                    min_c = min(p[1] for p in component)
                    max_c = max(p[1] for p in component)
                    area = (max_r - min_r + 1) * (max_c - min_c + 1)
                    num = len(component)
                    if num == area:
                        for x, y in component:
                            out[x][y] = 0
    return out