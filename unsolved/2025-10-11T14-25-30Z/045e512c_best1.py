from collections import deque
from typing import List, Tuple

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    # Helper masks
    ring_mask = [(0,0),(0,1),(0,2),(1,0),(1,2),(2,0),(2,1),(2,2)]  # center omitted
    plus_mask = [(0,1),(1,0),(1,1),(1,2),(2,1)]  # center + 4 neighbors

    # Utilities
    def in_bounds(r, c): return 0 <= r < R and 0 <= c < C
    def sign(x): 
        if x > 0: return 1
        if x < 0: return -1
        return 0

    # Detect template (ring preferred if present; else plus)
    template = None  # ('ring' or 'plus', color, top_left_r, top_left_c)
    for r0 in range(R - 2):
        for c0 in range(C - 2):
            # Try ring
            center = grid[r0+1][c0+1]
            # Ring requires center 0 and all 8 others same non-zero color
            if center == 0:
                col = grid[r0][c0]
                if col != 0:
                    ok = True
                    for dr, dc in ring_mask:
                        if grid[r0+dr][c0+dc] != col:
                            ok = False
                            break
                    if ok:
                        template = ('ring', col, r0, c0)
                        break
            # Try plus
            colp = grid[r0+1][c0+1]
            if colp != 0:
                okp = True
                for dr, dc in plus_mask:
                    if grid[r0+dr][c0+dc] != colp:
                        okp = False
                        break
                # Also ensure non-plus cells in this 3x3 are not that color (to be strict)
                if okp:
                    for dr in range(3):
                        for dc in range(3):
                            if (dr, dc) not in [(x[0], x[1]) for x in plus_mask]:
                                if (dr, dc) != (1,1) and grid[r0+dr][c0+dc] == colp:
                                    okp = False
                                    break
                        if not okp:
                            break
            if template is None and colp != 0 and okp:
                template = ('plus', colp, r0, c0)
                break
        if template is not None:
            break

    if template is None:
        # Fallback: nothing to do
        return out

    ttype, tcolor, t_r0, t_c0 = template
    t_center = (t_r0 + 1, t_c0 + 1)

    # Draw functions
    def draw_ring_at(r0: int, c0: int, color: int):
        if not (0 <= r0 <= R-3 and 0 <= c0 <= C-3):
            return
        for dr, dc in ring_mask:
            out[r0+dr][c0+dc] = color

    def draw_plus_center(r: int, c: int, color: int):
        if not (1 <= r <= R-2 and 1 <= c <= C-2):
            return
        out[r][c] = color
        out[r-1][c] = color
        out[r+1][c] = color
        out[r][c-1] = color
        out[r][c+1] = color

    # BFS to get components of all non-zero colors except the template color
    visited = [[False]*C for _ in range(R)]
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    components: List[Tuple[int, List[Tuple[int,int]]]] = []  # (color, list of cells)

    for r in range(R):
        for c in range(C):
            if visited[r][c]: 
                continue
            col = grid[r][c]
            if col == 0 or col == tcolor:
                visited[r][c] = True
                continue
            # BFS for this component
            q = deque([(r,c)])
            visited[r][c] = True
            cells = [(r,c)]
            while q:
                rr, cc = q.popleft()
                for dr, dc in dirs:
                    nr, nc = rr+dr, cc+dc
                    if in_bounds(nr, nc) and not visited[nr][nc] and grid[nr][nc] == col:
                        visited[nr][nc] = True
                        q.append((nr, nc))
                        cells.append((nr, nc))
            components.append((col, cells))

    # Helper to compute centroid (rounded)
    def centroid(cells: List[Tuple[int,int]]) -> Tuple[int,int]:
        s_r = sum(p[0] for p in cells) / len(cells)
        s_c = sum(p[1] for p in cells) / len(cells)
        # round to nearest int
        cr = int(round(s_r))
        cc = int(round(s_c))
        cr = max(0, min(R-1, cr))
        cc = max(0, min(C-1, cc))
        return cr, cc

    # Replication functions with "flush to border" logic
    def replicate_ring(anchor_r0: int, anchor_c0: int, dr: int, dc: int, color: int):
        # Clamp anchor to valid top-left
        r0 = max(0, min(R-3, anchor_r0))
        c0 = max(0, min(C-3, anchor_c0))
        draw_ring_at(r0, c0, color)
        last_r, last_c = r0, c0
        while True:
            nr = last_r + (dr*4 if dr != 0 else 0)
            nc = last_c + (dc*4 if dc != 0 else 0)
            if 0 <= nr <= R-3 and 0 <= nc <= C-3:
                draw_ring_at(nr, nc, color)
                last_r, last_c = nr, nc
            else:
                break
        # flush to the far border in the direction
        fr = last_r if dr == 0 else (0 if dr < 0 else R-3)
        fc = last_c if dc == 0 else (0 if dc < 0 else C-3)
        if (fr, fc) != (last_r, last_c):
            draw_ring_at(fr, fc, color)

    def replicate_plus(center_r: int, center_c: int, dr: int, dc: int, color: int):
        # Clamp center to valid
        r = max(1, min(R-2, center_r))
        c = max(1, min(C-2, center_c))
        draw_plus_center(r, c, color)
        last_r, last_c = r, c
        while True:
            nr = last_r + (dr*4 if dr != 0 else 0)
            nc = last_c + (dc*4 if dc != 0 else 0)
            if 1 <= nr <= R-2 and 1 <= nc <= C-2:
                draw_plus_center(nr, nc, color)
                last_r, last_c = nr, nc
            else:
                break
        # flush to border in direction
        fr = last_r if dr == 0 else (1 if dr < 0 else R-2)
        fc = last_c if dc == 0 else (1 if dc < 0 else C-2)
        if (fr, fc) != (last_r, last_c):
            draw_plus_center(fr, fc, color)

    # Apply replication per component
    for col, cells in components:
        # Compute direction from template center to component centroid
        cen_r, cen_c = centroid(cells)
        dr = sign(cen_r - t_center[0])
        dc = sign(cen_c - t_center[1])
        if dr == 0 and dc == 0:
            # If the centroid lies on the template center, nudge by looking at bbox
            minr = min(r for r,_ in cells); maxr = max(r for r,_ in cells)
            minc = min(c for _,c in cells); maxc = max(c for _,c in cells)
            dr = sign((minr+maxr)//2 - t_center[0])
            dc = sign((minc+maxc)//2 - t_center[1])
            if dr == 0 and dc == 0:
                # default push to the right
                dc = 1
        if ttype == 'ring':
            # anchor at component's top-left so its marks sit on the ring
            minr = min(r for r,_ in cells)
            minc = min(c for _,c in cells)
            replicate_ring(minr, minc, dr, dc, col)
        else:  # plus
            # anchor at rounded centroid as the plus center
            replicate_plus(cen_r, cen_c, dr, dc, col)

    return out