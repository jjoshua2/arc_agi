def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    grid = [row[:] for row in grid_lst]  # copy
    rows = len(grid)
    if rows == 0:
        return grid
    cols = len(grid[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0 and not visited[i][j]:
                # flood fill
                component = []
                stack = [(i, j)]
                visited[i][j] = True
                component.append((i, j))
                while stack:
                    x, y = stack.pop()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0 and not visited[nx][ny]:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
                            component.append((nx, ny))
                # check if internal
                is_internal = True
                for x, y in component:
                    if x == 0 or x == rows - 1 or y == 0 or y == cols - 1:
                        is_internal = False
                        break
                if is_internal:
                    for x, y in component:
                        grid[x][y] = 2
    return grid