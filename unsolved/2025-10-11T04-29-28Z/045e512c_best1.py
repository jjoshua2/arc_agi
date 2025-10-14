from collections import deque
from typing import List, Tuple
import math

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return [row[:] for row in grid]
    rows = len(grid)
    cols = len(grid[0])

    # Helper: 4-neighbor BFS components
    visited = [[False]*cols for _ in range(rows)]
    components = []  # list of (color, cells_list)
    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            if val != 0 and not visited[r][c]:
                color = val
                q = deque()
                q.append((r,c))
                visited[r][c] = True
                cells = []
                while q:
                    rr, cc = q.popleft()
                    cells.append((rr, cc))
                    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                        nr, nc = rr+dr, cc+dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            q.append((nr,nc))
                components.append((color, cells))

    if not components:
        return [row[:] for row in grid]

    # Choose the largest connected component as template S
    components.sort(key=lambda x: len(x[1]), reverse=True)
    S_color, S_cells = components[0]

    # S bounding box and mask
    min_r = min(r for r,_ in S_cells)
    max_r = max(r for r,_ in S_cells)
    min_c = min(c for _,c in S_cells)
    max_c = max(c for _,c in S_cells)
    h_S = max_r - min_r + 1
    w_S = max_c - min_c + 1
    # Relative mask positions inside bounding box where S_color appears
    S_mask = []
    S_cell_set = set(S_cells)
    for rr in range(min_r, max_r+1):
        for cc in range(min_c, max_c+1):
            if (rr,cc) in S_cell_set:
                S_mask.append((rr - min_r, cc - min_c))

    # Template center (integer)
    center_r = (min_r + max_r) // 2
    center_c = (min_c + max_c) // 2

    s = max(h_S, w_S)
    step = s + 1

    # Group all positions by color (excluding S_color and 0)
    color_positions = {}
    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            if val != 0 and val != S_color:
                color_positions.setdefault(val, []).append((r,c))

    output = [row[:] for row in grid]

    # Candidate direction unit vectors (8-neighbors)
    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    # Max k to try (safe upper bound)
    max_k = max(rows, cols) // max(1, step) + 4

    # Tolerance when matching predicted anchor to centroid
    tol = 0.8

    for color, positions in color_positions.items():
        # centroid
        mean_r = sum(p[0] for p in positions) / len(positions)
        mean_c = sum(p[1] for p in positions) / len(positions)
        best = None  # (distance, u_r, u_c, k)
        for (u_r, u_c) in dirs:
            # try k values
            for k in range(1, max_k+1):
                pred_anchor_r = center_r + u_r * (k*step - 1)
                pred_anchor_c = center_c + u_c * (k*step - 1)
                # distance to centroid
                d = math.hypot(pred_anchor_r - mean_r, pred_anchor_c - mean_c)
                if d <= tol:
                    # choose best match (smallest d)
                    if best is None or d < best[0]:
                        best = (d, u_r, u_c, k)
                    # we can break on first good k (but keep searching smaller d)
                # small optimization: if predicted anchor is already very far from centroid in Manhattan sense, skip bigger k
                # but keep simple
        if best is None:
            # No direction/k found: skip this color (not a marker)
            continue
        _, u_r, u_c, k_start = best

        # Place copies for n = k_start, k_start+1, ... until tile does not intersect grid
        n = k_start
        while True:
            center_copy_r = center_r + u_r * (n * step)
            center_copy_c = center_c + u_c * (n * step)
            top_r = center_copy_r - (h_S // 2)
            top_c = center_copy_c - (w_S // 2)
            bot_r = top_r + h_S - 1
            bot_c = top_c + w_S - 1
            # check if tile intersects grid at all
            if bot_r < 0 or bot_c < 0 or top_r >= rows or top_c >= cols:
                # fully outside grid; since n increases moving further out, we can break
                break
            # overlay S_mask recolored to "color"
            for (r_off, c_off) in S_mask:
                tr = top_r + r_off
                tc = top_c + c_off
                if 0 <= tr < rows and 0 <= tc < cols:
                    # preserve original template color cells
                    if output[tr][tc] == S_color:
                        continue
                    output[tr][tc] = color
            n += 1

    return output