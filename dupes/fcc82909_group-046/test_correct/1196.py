def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    h = len(grid_lst)
    if h == 0:
        return []
    w = len(grid_lst[0])
    input_grid = grid_lst
    output = [row[:] for row in input_grid]
    visited = [[False for _ in range(w)] for _ in range(h)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(h):
        for j in range(w):
            if input_grid[i][j] > 0 and not visited[i][j]:
                stack = [(i, j)]
                visited[i][j] = True
                colors_set = set()
                min_c = j
                max_c = j
                max_r = i
                stack.append((i, j))  # Ensure first is processed in loop
                while stack:
                    x, y = stack.pop()
                    colors_set.add(input_grid[x][y])
                    min_c = min(min_c, y)
                    max_c = max(max_c, y)
                    max_r = max(max_r, x)
                    for dx, dy in directions:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < h and 0 <= ny < w and input_grid[nx][ny] > 0 and not visited[nx][ny]:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
                N = len(colors_set)
                for k in range(1, N + 1):
                    nr = max_r + k
                    if nr >= h:
                        break
                    for c in range(min_c, max_c + 1):
                        if output[nr][c] == 0:
                            output[nr][c] = 3
    return output