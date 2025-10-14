def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid:
        return []
    h = len(grid)
    if h == 0:
        return []
    w = len(grid[0])
    output = [row[:] for row in grid]
    visited = [[False] * w for _ in range(h)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connected

    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0 and not visited[r][c]:
                color = grid[r][c]
                component = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    x, y = stack.pop()
                    component.append((x, y))
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and not visited[nx][ny] and grid[nx][ny] == color:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
                if not component:
                    continue
                min_r = min(x for x, _ in component)
                max_r = max(x for x, _ in component)
                min_c = min(y for _, y in component)
                max_c = max(y for _, y in component)
                height = max_r - min_r + 1
                width = max_c - min_c + 1
                if height < width:
                    # fill vertical lines
                    cols_to_fill = set()
                    for i in range(min_r, max_r + 1):
                        for j in range(min_c, max_c + 1):
                            if grid[i][j] == 0:
                                cols_to_fill.add(j)
                    for j in cols_to_fill:
                        for i in range(min_r, max_r + 1):
                            output[i][j] = 0
                elif height > width:
                    # fill horizontal lines
                    rows_to_fill = set()
                    for i in range(min_r, max_r + 1):
                        for j in range(min_c, max_c + 1):
                            if grid[i][j] == 0:
                                rows_to_fill.add(i)
                    for i in rows_to_fill:
                        for j in range(min_c, max_c + 1):
                            output[i][j] = 0
    return output