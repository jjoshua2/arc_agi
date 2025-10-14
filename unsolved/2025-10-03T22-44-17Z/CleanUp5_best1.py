def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid:
        return []
    h, w = len(grid), len(grid[0])
    new_grid = [row[:] for row in grid]
    visited = [[False] * w for _ in range(h)]
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for i in range(h):
        for j in range(w):
            if new_grid[i][j] != 0 and not visited[i][j]:
                color = new_grid[i][j]
                component = []
                stack = [(i, j)]
                visited[i][j] = True
                size = 0
                while stack:
                    x, y = stack.pop()
                    component.append((x, y))
                    size += 1
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and not visited[nx][ny] and new_grid[nx][ny] == color:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
                if size == 1:
                    for x, y in component:
                        new_grid[x][y] = 0
    return new_grid