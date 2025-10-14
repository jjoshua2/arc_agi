def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows, cols = len(grid), len(grid[0])
    output = [row[:] for row in grid]
    visited = [[False] * cols for _ in range(rows)]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def dfs(i: int, j: int, component: list[tuple[int, int]]) -> None:
        stack = [(i, j)]
        visited[i][j] = True
        component.append((i, j))
        while stack:
            x, y = stack.pop()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and output[nx][ny] == 5:
                    visited[nx][ny] = True
                    component.append((nx, ny))
                    stack.append((nx, ny))

    for i in range(rows):
        for j in range(cols):
            if output[i][j] == 5 and not visited[i][j]:
                component = []
                dfs(i, j, component)
                if not component:
                    continue
                rs = [p[0] for p in component]
                cs = [p[1] for p in component]
                min_r, max_r = min(rs), max(rs)
                min_c, max_c = min(cs), max(cs)
                height = max_r - min_r + 1
                width = max_c - min_c + 1
                L = len(component)
                if width > height:
                    # horizontal
                    if L <= 4:
                        color = 2
                    elif L == 5:
                        color = 4
                    else:
                        color = 1
                elif height > width:
                    # vertical
                    if L <= 3:
                        color = 2
                    elif L <= 5:
                        color = 4
                    else:
                        color = 1
                else:
                    # square or equal, default to 2 (assuming no such cases)
                    color = 2
                for x, y in component:
                    output[x][y] = color
    return output