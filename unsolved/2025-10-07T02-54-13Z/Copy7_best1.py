import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    components = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connected

    for i in range(rows):
        for j in range(cols):
            if grid[i, j] > 0 and not visited[i, j]:
                component_color = grid[i, j]
                comp_cells = []
                stack = [(i, j)]
                min_r, max_r = i, i
                min_c, max_c = j, j
                while stack:
                    x, y = stack.pop()
                    if visited[x, y]:
                        continue
                    visited[x, y] = True
                    comp_cells.append((x, y))
                    min_r = min(min_r, x)
                    max_r = max(max_r, x)
                    min_c = min(min_c, y)
                    max_c = max(max_c, y)
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny] and grid[nx, ny] == component_color:
                            stack.append((nx, ny))
                h = max_r - min_r + 1
                w = max_c - min_c + 1
                components.append({
                    'color': component_color,
                    'min_r': min_r, 'max_r': max_r,
                    'min_c': min_c, 'max_c': max_c,
                    'h': h, 'w': w
                })

    # Find source and pattern
    source = None
    pattern = {}  # local_r -> set of local_c
    outlier_color = None
    for comp in components:
        outlier_cs = {}
        has_outlier = False
        for r in range(comp['min_r'], comp['max_r'] + 1):
            for c in range(comp['min_c'], comp['max_c'] + 1):
                if grid[r, c] > 0 and grid[r, c] != comp['color']:
                    local_r = r - comp['min_r']
                    local_c = c - comp['min_c']
                    if local_r not in outlier_cs:
                        outlier_cs[local_r] = set()
                    outlier_cs[local_r].add(local_c)
                    has_outlier = True
                    if outlier_color is None:
                        outlier_color = grid[r, c]
                    # Assume all outliers same color
        if has_outlier:
            source = comp
            pattern = outlier_cs
            break  # Assume one source

    if source is None:
        return grid.tolist()

    # Apply pattern to all components
    output = grid.copy()
    h_s = source['h']
    for comp in components:
        h_t = comp['h']
        w_t = comp['w']
        minr_t = comp['min_r']
        minc_t = comp['min_c']
        for i in range(h_t):
            s_r = h_s - h_t + i
            if 0 <= s_r < h_s and s_r in pattern:
                for loc_c in pattern[s_r]:
                    if loc_c < w_t:
                        t_r = minr_t + i
                        t_c = minc_t + loc_c
                        output[t_r, t_c] = outlier_color

    return output.tolist()