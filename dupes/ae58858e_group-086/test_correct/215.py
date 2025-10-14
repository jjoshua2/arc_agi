def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid:
        return []
    h = len(grid)
    if h == 0:
        return []
    w = len(grid[0])
    out = [row[:] for row in grid]
    visited = [[False] * w for _ in range(h)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(h):
        for j in range(w):
            if out[i][j] == 2 and not visited[i][j]:
                component = []
                stack = [(i, j)]
                visited[i][j] = True
                while stack:
                    x, y = stack.pop()
                    component.append((x, y))
                    for dx, dy in directions:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < h and 0 <= ny < w and not visited[nx][ny] and out[nx][ny] == 2:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
                size = len(component)
                if size >= 4:
                    for x, y in component:
                        out[x][y] = 6
    return out