import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    background = grid[0, 0]

    visited = np.zeros((rows, cols), bool)
    components = []
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] != background and not visited[i, j]:
                component = []
                colors_count = {}
                stack = [(i, j)]
                visited[i, j] = True
                while stack:
                    r, c = stack.pop()
                    component.append((r, c))
                    colr = grid[r, c]
                    colors_count[colr] = colors_count.get(colr, 0) + 1
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != background and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                components.append({'cells': component, 'size': len(component), 'colors_count': colors_count})

    components.sort(key=lambda x: x['size'])
    template_comp = components[0]
    center_color = min(template_comp['colors_count'], key=template_comp['colors_count'].get)
    centers = [(r, c) for r, c in template_comp['cells'] if grid[r, c] == center_color]
    cr, cc = centers[0]
    template_rel = [(r - cr, c - cc, grid[r, c]) for r, c in template_comp['cells']]

    out = grid.copy()
    for r, c in template_comp['cells']:
        out[r, c] = background

    anchors = []
    for i in range(rows):
        for j in range(cols):
            if out[i, j] == center_color:
                anchors.append((i, j))

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for ar, ac in anchors:
        for dr, dc in directions:
            ray_positions = {}
            for tr, tc, tcol in template_rel:
                if (dc == 0 and tc == 0 and tr * dr > 0 and tr % dr == 0):
                    k = tr // dr
                    ray_positions[k] = tcol
                elif (dr == 0 and tr == 0 and tc * dc > 0 and tc % dc == 0):
                    k = tc // dc
                    ray_positions[k] = tcol
            if ray_positions:
                max_k = max(ray_positions.keys())
                if max_k >= 2:
                    ext_color = ray_positions[max_k]
                    k = max_k + 1
                    while True:
                        nr = ar + k * dr
                        nc = ac + k * dc
                        if not (0 <= nr < rows and 0 <= nc < cols) or grid[nr, nc] == background:
                            break
                        out[nr, nc] = ext_color
                        k += 1

    for ar, ac in anchors:
        for dr, dc, colr in template_rel:
            nr = ar + dr
            nc = ac + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                out[nr, nc] = colr

    return out.tolist()