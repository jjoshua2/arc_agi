def get_components(grid):
    h = len(grid)
    if h == 0:
        return []
    w = len(grid[0])
    visited = [[False] * w for _ in range(h)]
    components = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(h):
        for j in range(w):
            if grid[i][j] != 0 and not visited[i][j]:
                color = grid[i][j]
                component = []
                stack = [(i, j)]
                visited[i][j] = True
                while stack:
                    x, y = stack.pop()
                    component.append((x, y))
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and not visited[nx][ny] and grid[nx][ny] == color:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
                components.append((color, component))
    return components

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return grid_lst
    h = len(grid_lst)
    w = len(grid_lst[0])
    output = [[0] * w for _ in range(h)]
    all_comps = get_components(grid_lst)
    for color, comp in all_comps:
        if not comp:
            continue
        rows = [r for r, c in comp]
        cols = [c for r, c in comp]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        bh = max_r - min_r + 1
        bw = max_c - min_c + 1
        if bh == 3 and bw in (2, 3):
            for r, c in comp:
                output[r][c] = color
    return output