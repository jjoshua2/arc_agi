def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    n_rows = len(grid_lst)
    if n_rows == 0:
        return grid_lst
    n_cols = len(grid_lst[0])
    grid = grid_lst  # for reference

    def dfs(r, c, visited):
        stack = [(r, c)]
        component = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while stack:
            cr, cc = stack.pop()
            if (0 <= cr < n_rows and 0 <= cc < n_cols and grid[cr][cc] != 0 and not visited[cr][cc]):
                visited[cr][cc] = True
                component.append((cr, cc))
                for dr, dc in directions:
                    stack.append((cr + dr, cc + dc))
        return component

    visited = [[False] * n_cols for _ in range(n_rows)]
    components = []
    for i in range(n_rows):
        for j in range(n_cols):
            if grid[i][j] != 0 and not visited[i][j]:
                comp = dfs(i, j, visited)
                components.append(comp)

    if len(components) != 2:
        return [row[:] for row in grid_lst]  # fallback, but assume correct

    pattern_comp = None
    target_comp = None
    for comp in components:
        colors = {}
        for r, c in comp:
            col = grid[r][c]
            colors[col] = colors.get(col, 0) + 1
        if len(colors) > 1:
            primary = max(colors, key=colors.get)
            secondary = next(k for k in colors if k != primary)
            min_r = min(r for r, c in comp)
            max_r = max(r for r, c in comp)
            min_c = min(c for r, c in comp)
            max_c = max(c for r, c in comp)
            pattern_comp = {
                'positions': comp,
                'primary': primary,
                'secondary': secondary,
                'min_r': min_r,
                'max_r': max_r,
                'min_c': min_c,
                'max_c': max_c
            }
        else:
            primary = next(iter(colors))
            min_r = min(r for r, c in comp)
            max_r = max(r for r, c in comp)
            min_c = min(c for r, c in comp)
            max_c = max(c for r, c in comp)
            target_comp = {
                'positions': comp,
                'primary': primary,
                'min_r': min_r,
                'max_r': max_r,
                'min_c': min_c,
                'max_c': max_c
            }

    if pattern_comp is None or target_comp is None:
        return [row[:] for row in grid_lst]

    p = pattern_comp
    t = target_comp
    row_overlap = max(p['min_r'], t['min_r']) <= min(p['max_r'], t['max_r'])
    col_overlap = max(p['min_c'], t['min_c']) <= min(p['max_c'], t['max_c'])

    output = [row[:] for row in grid_lst]

    if row_overlap and not col_overlap:
        # horizontal
        shift_c = t['min_c'] - p['min_c']
        for r, c in p['positions']:
            if grid[r][c] == p['secondary']:
                new_c = c + shift_c
                if 0 <= new_c < n_cols:
                    output[r][new_c] = p['secondary']
    elif col_overlap and not row_overlap:
        # vertical
        shift_r = t['min_r'] - p['min_r']
        for r, c in p['positions']:
            if grid[r][c] == p['secondary']:
                new_r = r + shift_r
                if 0 <= new_r < n_rows:
                    output[new_r][c] = p['secondary']

    return output