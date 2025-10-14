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

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    grid = [row[:] for row in grid_lst]  # mutable copy
    h = len(grid)
    w = len(grid[0])
    if h != 20 or w != 20:
        return grid  # Assume 20x20, but handle generally
    comps = get_components(grid)
    multi = [comp for comp in comps if len(comp[1]) > 1]
    if len(multi) != 1:
        return grid  # Assume exactly one
    shape_color = multi[0][0]
    shape_pos = multi[0][1]
    single_comps = [comp for comp in comps if len(comp[1]) == 1]
    single_colors = set(comp[0] for comp in single_comps)
    if len(single_colors) != 1 or list(single_colors)[0] == shape_color:
        return grid  # Assume exactly one ink color different from shape
    ink_color = list(single_colors)[0]
    single_ink = []
    for color, pos_list in single_comps:
        if color == ink_color:
            single_ink.extend(pos_list)
    # Compute max row per column for shape
    max_row_per_col = [-1] * w
    for r, c in shape_pos:
        max_row_per_col[c] = max(max_row_per_col[c], r)
    # For each relevant column
    for c in range(w):
        if max_row_per_col[c] != -1:
            r_bottom = max_row_per_col[c]
            has_below = any(sr > r_bottom and sc == c for sr, sc in single_ink)
            if has_below:
                for r in range(r_bottom + 1, h):
                    if grid[r][c] == 0:
                        grid[r][c] = ink_color
    return grid