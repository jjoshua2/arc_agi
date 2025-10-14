import numpy as np
from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # Find largest 2 component
    max_size = 0
    min_r_large = rows
    max_r_large = -1
    min_c_large = cols
    max_c_large = -1
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 2 and not visited[r, c]:
                q = deque([(r, c)])
                visited[r, c] = True
                local_min_r = local_max_r = r
                local_min_c = local_max_c = c
                size = 1
                while q:
                    cr, cc = q.popleft()
                    local_min_r = min(local_min_r, cr)
                    local_max_r = max(local_max_r, cr)
                    local_min_c = min(local_min_c, cc)
                    local_max_c = max(local_max_c, cc)
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and grid[nr, nc] == 2:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                            size += 1
                if size > max_size:
                    max_size = size
                    min_r_large = local_min_r
                    max_r_large = local_max_r
                    min_c_large = local_min_c
                    max_c_large = local_max_c
    h = max_r_large - min_r_large + 1 if max_size > 0 else 0
    w = max_c_large - min_c_large + 1 if max_size > 0 else 0
    if h == 0 or w == 0:
        return []
    out = np.full((h, w), 2, dtype=int)

    # Reset visited
    visited = np.zeros((rows, cols), dtype=bool)

    # Find small 3x3 non-0 components
    smalls = []
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0 and not visited[r, c]:
                q = deque([(r, c)])
                visited[r, c] = True
                local_min_r = local_max_r = r
                local_min_c = local_max_c = c
                size = 1
                while q:
                    cr, cc = q.popleft()
                    local_min_r = min(local_min_r, cr)
                    local_max_r = max(local_max_r, cr)
                    local_min_c = min(local_min_c, cc)
                    local_max_c = max(local_max_c, cc)
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and grid[nr, nc] != 0:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                            size += 1
                comp_h = local_max_r - local_min_r + 1
                comp_w = local_max_c - local_min_c + 1
                if comp_h == 3 and comp_w == 3 and size == 9:
                    # Check if outside large
                    if local_max_r < min_r_large or local_min_r > max_r_large or local_max_c < min_c_large or local_min_c > max_c_large:
                        sub = grid[local_min_r:local_min_r+3, local_min_c:local_min_c+3].copy()
                        smalls.append((local_min_r, local_min_c, sub))

    # Sort by min_r, min_c
    smalls.sort(key=lambda x: (x[0], x[1]))

    # Overlay
    for i, (minr, minc, sub) in enumerate(smalls):
        offset_r = 1 if h % 2 == 0 else 2
        start_r = max(0, min(h - 3, offset_r + i * 3))
        offset_c = 1 if w % 2 == 0 else 2
        start_c = max(0, min(w - 3, offset_c + i * 3))
        for ii in range(3):
            for jj in range(3):
                nr = start_r + ii
                nc = start_c + jj
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = sub[ii, jj]

    return out.tolist()