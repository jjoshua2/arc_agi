import numpy as np

def get_components(grid):
    h = len(grid)
    w = len(grid[0])
    visited = np.zeros((h, w), dtype=bool)
    components = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connected
    for i in range(h):
        for j in range(w):
            if grid[i][j] > 0 and not visited[i][j]:
                color = grid[i][j]
                stack = [(i, j)]
                pos = []
                visited[i][j] = True
                while stack:
                    x, y = stack.pop()
                    pos.append((x, y))
                    for dx, dy in directions:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < h and 0 <= ny < w and not visited[nx][ny] and grid[nx][ny] == color:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
                components.append((color, pos))
    return components

def get_shape(pos):
    if not pos:
        return frozenset()
    minr = min(r for r, c in pos)
    minc = min(c for r, c in pos)
    rel = frozenset((r - minr, c - minc) for r, c in pos)
    return rel

def transform(grid_lst):
    if not grid_lst:
        return []
    grid = [row[:] for row in grid_lst]  # mutable copy
    comps = get_components(grid)
    shape_to_color = {}
    one_comps = []  # list of (shape, pos)
    for color, pos in comps:
        shape = get_shape(pos)
        if color == 1:
            one_comps.append((shape, pos))
        else:
            # Assume unique shapes for non-1 components
            shape_to_color[shape] = color
    # Recolor 1-components
    for shape, pos in one_comps:
        if shape in shape_to_color:
            new_color = shape_to_color[shape]
            for r, c in pos:
                grid[r][c] = new_color
    return grid