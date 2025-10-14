def transform(input_grid):
    if not input_grid or not input_grid[0]:
        return input_grid
    m = len(input_grid)
    n = len(input_grid[0])
    grid = [row[:] for row in input_grid]
    visited = [[False] * n for _ in range(m)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def dfs(i, j, component):
        stack = [(i, j)]
        visited[i][j] = True
        component.append((i, j))
        while stack:
            x, y = stack.pop()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and not visited[nx][ny] and grid[nx][ny] == 5:
                    visited[nx][ny] = True
                    component.append((nx, ny))
                    stack.append((nx, ny))

    for i in range(m):
        for j in range(n):
            if grid[i][j] == 5 and not visited[i][j]:
                component = []
                dfs(i, j, component)
                size = len(component)
                color = 2 if size == 6 else 1
                for x, y in component:
                    grid[x][y] = color
    return grid