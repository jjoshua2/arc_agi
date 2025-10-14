import numpy as np

def get_components(subgrid, target_val=None):
    if subgrid.size == 0:
        return []
    visited = np.zeros(subgrid.shape, dtype=bool)
    components = []
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(subgrid.shape[0]):
        for j in range(subgrid.shape[1]):
            if not visited[i, j]:
                val = subgrid[i, j]
                if target_val is not None:
                    if val != target_val:
                        continue
                    color = target_val
                else:
                    if val == 0:
                        continue
                    color = val
                comp = []
                stack = [(i, j)]
                visited[i, j] = True
                while stack:
                    x, y = stack.pop()
                    comp.append((x, y))
                    for dx, dy in dirs:
                        nx = x + dx
                        ny = y + dy
                        if (0 <= nx < subgrid.shape[0] and
                            0 <= ny < subgrid.shape[1] and
                            not visited[nx, ny] and
                            subgrid[nx, ny] == color):
                            visited[nx, ny] = True
                            stack.append((nx, ny))
                if comp:
                    components.append((color, comp))
    return components

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    half = cols // 2
    left = grid[:, :half]
    right = grid[:, half:]

    hole_comps = get_components(left, target_val=0)
    shape_dict = {}
    for color, comp in get_components(right):
        if not comp:
            continue
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        min_r = min(rs)
        min_c = min(cs)
        pattern = frozenset((r - min_r, c - min_c) for r, c in comp)
        key = (min_r, pattern)
        shape_dict[key] = color

    output_grid = left.copy()
    for _, comp in hole_comps:
        if not comp:
            continue
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        min_r = min(rs)
        min_c = min(cs)
        pattern = frozenset((r - min_r, c - min_c) for r, c in comp)
        key = (min_r, pattern)
        if key in shape_dict:
            fill_color = shape_dict[key]
            for r, c in comp:
                output_grid[r, c] = fill_color

    return output_grid.tolist()