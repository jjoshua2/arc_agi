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

def normalize_shape(component):
    if not component:
        return frozenset()
    min_r = min(r for r, c in component)
    min_c = min(c for r, c in component)
    shape = frozenset((r - min_r, c - min_c) for r, c in component)
    return shape

def transform(grid_lst):
    grid = [row[:] for row in grid_lst]
    all_comps = get_components(grid)
    one_comps = [comp for color, comp in all_comps if color == 1]
    other_comps = [(color, comp) for color, comp in all_comps if color != 1]
    shape_to_color = {}
    for color, comp in other_comps:
        shape = normalize_shape(comp)
        shape_to_color[shape] = color
    for comp in one_comps:
        shape = normalize_shape(comp)
        if shape in shape_to_color:
            new_color = shape_to_color[shape]
            for r, c in comp:
                grid[r][c] = new_color
    return grid