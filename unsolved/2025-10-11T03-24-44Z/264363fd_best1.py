def transform(grid):
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])
    original = [row[:] for row in grid]
    b = grid[0][0]  # assume all border cells have the same color b
    visited = [[False for _ in range(w)] for _ in range(h)]
    components = []
    for i in range(h):
        for j in range(w):
            if not visited[i][j] and grid[i][j] != b:
                component = []
                stack = [(i, j)]
                visited[i][j] = True
                c = grid[i][j]
                component.append((i, j))
                while stack:
                    x, y = stack.pop()
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < h and 0 <= ny < w and not visited[nx][ny] and grid[nx][ny] == c:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
                            component.append((nx, ny))
                components.append((c, component))
    for c_comp, component in components:
        if len(component) <= 20:
            for r_pos, c_pos in component:
                grid[r_pos][c_pos] = b
        else:
            component_set = set(component)
            for r, col in component:
                if grid[r][col] > c_comp:
                    k = grid[r][col]
                    diff = k - c_comp
                    i_val = 3 - diff
                    a = c_comp + i_val
                    d = a + (1 if (c_comp == 1 or diff == 1) else 0)
                    # set the row
                    for pr, pc in component:
                        if pr == r and original[pr][pc] == c_comp:
                            grid[pr][pc] = max(grid[pr][pc], a)
                    # set the column
                    for pr, pc in component:
                        if pc == col and original[pr][pc] == c_comp:
                            grid[pr][pc] = max(grid[pr][pc], a)
                    # set the diagonals
                    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        nr = r + dx
                        nc = col + dy
                        if 0 <= nr < h and 0 <= nc < w:
                            found = (nr, nc) in component_set
                            if found and original[nr][nc] == c_comp:
                                grid[nr][nc] = max(grid[nr][nc], d)
    return grid