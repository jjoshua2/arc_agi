def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])
    output = [row[:] for row in grid]
    visited = [[False] * w for _ in range(h)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def dfs(i, j, component):
        stack = [(i, j)]
        visited[i][j] = True
        component.append((i, j))
        while stack:
            x, y = stack.pop()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and not visited[nx][ny] and grid[nx][ny] == 0:
                    visited[nx][ny] = True
                    stack.append((nx, ny))
                    component.append((nx, ny))

    for i in range(h):
        for j in range(w):
            if grid[i][j] == 0 and not visited[i][j]:
                component = []
                dfs(i, j, component)
                borders = set()
                for x, y in component:
                    if x == 0:
                        borders.add('top')
                    if x == h - 1:
                        borders.add('bottom')
                    if y == 0:
                        borders.add('left')
                    if y == w - 1:
                        borders.add('right')
                if len(borders) <= 1:
                    for x, y in component:
                        output[x][y] = 4
    return output