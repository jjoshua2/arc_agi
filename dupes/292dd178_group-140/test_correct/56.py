def find_components(grid):
    h = len(grid)
    w = len(grid[0])
    visited = [[False] * w for _ in range(h)]
    components = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(h):
        for j in range(w):
            if grid[i][j] == 1 and not visited[i][j]:
                comp = []
                stack = [(i, j)]
                visited[i][j] = True
                while stack:
                    x, y = stack.pop()
                    comp.append((x, y))
                    for dx, dy in directions:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < h and 0 <= ny < w and grid[nx][ny] == 1 and not visited[nx][ny]:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
                components.append(comp)
    return components

def transform(grid_lst):
    if not grid_lst:
        return []
    grid = [row[:] for row in grid_lst]
    h = len(grid)
    w = len(grid[0])
    if h == 0 or w == 0:
        return grid
    B = grid[0][0]
    if B == 1:
        return grid
    comps = find_components(grid)
    for comp in comps:
        if not comp:
            continue
        comp_set = set(comp)
        # Vertical fills
        spanned_cols = set(c for _, c in comp)
        for c in spanned_cols:
            rows_1 = sorted(r for r, cc in comp if cc == c)
            m = len(rows_1)
            if m >= 2:
                for jj in range(m - 1):
                    st = rows_1[jj] + 1
                    en = rows_1[jj + 1] - 1
                    for rr in range(st, en + 1):
                        if 0 <= rr < h and grid[rr][c] == B:
                            grid[rr][c] = 2
            if m == 1:
                q = rows_1[0]
                # Extend up
                extend_u = False
                adj_cs = []
                if c > 0:
                    adj_cs.append(c - 1)
                if c < w - 1:
                    adj_cs.append(c + 1)
                for ac in adj_cs:
                    for rr in range(q):
                        if (rr, ac) in comp_set:
                            extend_u = True
                            break
                    if extend_u:
                        break
                if extend_u:
                    for rr in range(q):
                        if grid[rr][c] == B:
                            grid[rr][c] = 2
                # Extend down
                extend_d = False
                for ac in adj_cs:
                    for rr in range(q + 1, h):
                        if (rr, ac) in comp_set:
                            extend_d = True
                            break
                    if extend_d:
                        break
                if extend_d:
                    for rr in range(q + 1, h):
                        if grid[rr][c] == B:
                            grid[rr][c] = 2
        # Horizontal fills
        spanned_rows = set(r for r, _ in comp)
        for r in spanned_rows:
            cols_1 = sorted(c for rr, c in comp if rr == r)
            k = len(cols_1)
            if k >= 2:
                for ii in range(k - 1):
                    st = cols_1[ii] + 1
                    en = cols_1[ii + 1] - 1
                    for cc in range(st, en + 1):
                        if 0 <= cc < w and grid[r][cc] == B:
                            grid[r][cc] = 2
            if k == 1:
                p = cols_1[0]
                # Extend left
                extend_l = False
                adj_rs = []
                if r > 0:
                    adj_rs.append(r - 1)
                if r < h - 1:
                    adj_rs.append(r + 1)
                for ar in adj_rs:
                    for cc in range(p):
                        if (ar, cc) in comp_set:
                            extend_l = True
                            break
                    if extend_l:
                        break
                if extend_l:
                    for cc in range(p):
                        if grid[r][cc] == B:
                            grid[r][cc] = 2
                # Extend right
                extend_r = False
                for ar in adj_rs:
                    for cc in range(p + 1, w):
                        if (ar, cc) in comp_set:
                            extend_r = True
                            break
                    if extend_r:
                        break
                if extend_r:
                    for cc in range(p + 1, w):
                        if grid[r][cc] == B:
                            grid[r][cc] = 2
    return grid