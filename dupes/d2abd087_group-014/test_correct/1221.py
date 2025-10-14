def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    output = [row[:] for row in grid_lst]
    visited = [[False] * cols for _ in range(rows)]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for i in range(rows):
        for j in range(cols):
            if grid_lst[i][j] == 5 and not visited[i][j]:
                component = []
                stack = [(i, j)]
                visited[i][j] = True
                while stack:
                    x, y = stack.pop()
                    component.append((x, y))
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and grid_lst[nx][ny] == 5:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
                size = len(component)
                color = 2 if size == 6 else 1
                for x, y in component:
                    output[x][y] = color
    return output