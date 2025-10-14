import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    valid_comps = []
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 2 and not visited[r, c]:
                comp = []
                stack = [(r, c)]
                visited[r, c] = True
                while stack:
                    x, y = stack.pop()
                    comp.append((x, y))
                    for dx, dy in dirs:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny] and grid[nx, ny] == 2:
                            visited[nx, ny] = True
                            stack.append((nx, ny))
                if comp:
                    minr = min(ii for ii, jj in comp)
                    maxr = max(ii for ii, jj in comp)
                    minc = min(jj for ii, jj in comp)
                    maxc = max(jj for ii, jj in comp)
                    v = len(comp)
                    bb = (maxr - minr + 1) * (maxc - minc + 1)
                    if v != bb:
                        valid_comps.append((maxr, maxc, comp))

    if not valid_comps:
        return []

    # Select the one with max maxr, then max maxc
    valid_comps.sort(key=lambda x: (-x[0], -x[1]))
    selected_comp = valid_comps[0][2]

    minr = min(ii for ii, jj in selected_comp)
    maxr = max(ii for ii, jj in selected_comp)
    minc = min(jj for ii, jj in selected_comp)
    maxc = max(jj for ii, jj in selected_comp)
    h = maxr - minr + 1
    w = maxc - minc + 1
    s = max(h, w) + 2
    out_grid = np.full((s, s), 8, dtype=int)
    for i, j in selected_comp:
        out_i = i - minr + 1
        out_j = j - minc + 1
        out_grid[out_i, out_j] = 2

    return out_grid.tolist()