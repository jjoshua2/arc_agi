def get_two_components(grid):
    h = len(grid)
    w = len(grid[0])
    visited = [[False] * w for _ in range(h)]
    components = []
    directions = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)]
    for i in range(h):
        for j in range(w):
            if grid[i][j] == 2 and not visited[i][j]:
                component = []
                stack = [(i, j)]
                visited[i][j] = True
                while stack:
                    x, y = stack.pop()
                    component.append((x, y))
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and not visited[nx][ny] and grid[nx][ny] == 2:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
                components.append(component)
    return components

def transform(grid_lst):
    if not grid_lst:
        return []
    h = len(grid_lst)
    w = len(grid_lst[0])
    two_comps = get_two_components(grid_lst)
    out = [row[:] for row in grid_lst]
    for comp in two_comps:
        if not comp:
            continue
        min_r = min(r for r, c in comp)
        min_c = min(c for r, c in comp)
        shape = frozenset((r - min_r, c - min_c) for r, c in comp)
        for i in range(h):
            for j in range(w):
                match = True
                positions = []
                for dr, dc in shape:
                    nr = i + dr
                    nc = j + dc
                    if not (0 <= nr < h and 0 <= nc < w and grid_lst[nr][nc] == 0):
                        match = False
                        break
                    positions.append((nr, nc))
                if match:
                    for pr, pc in positions:
                        out[pr][pc] = 2
    return out