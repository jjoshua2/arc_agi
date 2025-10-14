from collections import deque
from typing import List, Tuple

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    h, w = len(grid), len(grid[0])

    # 4-adjacent connected components of non-zero cells
    visited = [[False]*w for _ in range(h)]
    comps = []  # list of dicts: {'color': c, 'cells': [(r,c), ...]}
    dirs4 = [(1,0),(-1,0),(0,1),(0,-1)]

    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0 and not visited[r][c]:
                color = grid[r][c]
                q = deque([(r,c)])
                visited[r][c] = True
                cells = [(r,c)]
                while q:
                    rr, cc = q.popleft()
                    for dr, dc in dirs4:
                        nr, nc = rr+dr, cc+dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            q.append((nr,nc))
                            cells.append((nr,nc))
                comps.append({'color': color, 'cells': cells})

    if not comps:
        return [row[:] for row in grid]

    # Choose template: the largest component
    template_comp = max(comps, key=lambda d: len(d['cells']))
    template_color = template_comp['color']
    t_cells = template_comp['cells']

    # Template center: center of 3x3 bounding box
    trs = [r for r,_ in t_cells]
    tcs = [c for _,c in t_cells]
    rmin, rmax = min(trs), max(trs)
    cmin, cmax = min(tcs), max(tcs)

    # Ensure we have a 3x3 neighborhood around a center
    # Center is the middle of bounding box (works for 3x3 templates in tasks)
    t_center_r = (rmin + rmax) // 2
    t_center_c = (cmin + cmax) // 2

    # Build 3x3 mask (True where template occupies within that 3x3, False otherwise)
    mask = [[False]*3 for _ in range(3)]
    r0 = t_center_r - 1
    c0 = t_center_c - 1
    t_set = set(t_cells)
    for i in range(3):
        for j in range(3):
            rr, cc = r0 + i, c0 + j
            if (rr, cc) in t_set:
                mask[i][j] = True

    # Helper: sign function
    def sgn(x: float) -> int:
        if x > 0: return 1
        if x < 0: return -1
        return 0

    # Prepare output as copy of input
    out = [row[:] for row in grid]

    # For each other component, compute direction and stamp template mask repeatedly
    for comp in comps:
        if comp is template_comp:
            continue
        color = comp['color']
        # Compute component "center" by averaging coordinates (works well for short fragments)
        rs = [r for r,_ in comp['cells']]
        cs = [c for _,c in comp['cells']]
        sr = round(sum(rs)/len(rs))
        sc = round(sum(cs)/len(cs))
        dr = sgn(sr - t_center_r)
        dc = sgn(sc - t_center_c)
        if dr == 0 and dc == 0:
            continue  # shouldn't happen (that's the template)
        # Step vector is 4 units in the chosen direction (pattern size 3 + gap 1)
        step_r, step_c = 4*dr, 4*dc

        # Start from the first copy away from the template
        cr, cc = t_center_r + step_r, t_center_c + step_c

        # Stamp until completely outside
        while True:
            # If the 3x3 window is completely outside, stop
            if cr + 1 < 0 or cr - 1 >= h or cc + 1 < 0 or cc - 1 >= w:
                break
            # Place mask (onto zeros only), clipped to grid
            for i in range(3):
                for j in range(3):
                    if not mask[i][j]:
                        continue
                    rr, cc2 = cr + (i-1), cc + (j-1)
                    if 0 <= rr < h and 0 <= cc2 < w and out[rr][cc2] == 0:
                        out[rr][cc2] = color
            # Advance
            cr += step_r
            cc += step_c

    return out