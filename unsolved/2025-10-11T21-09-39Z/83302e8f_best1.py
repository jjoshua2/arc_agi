from collections import deque, Counter

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    h = len(grid)
    w = len(grid[0])

    # Count colors; choose the barrier color as the most frequent non-zero color
    counts = Counter()
    for r in range(h):
        for c in range(w):
            counts[grid[r][c]] += 1

    # Exclude background 0 when selecting barrier; if all are 0, nothing to do
    candidates = [(col, cnt) for col, cnt in counts.items() if col != 0]
    if not candidates:
        return [row[:] for row in grid]

    # Choose the most frequent non-zero color as barrier
    barrier_color = max(candidates, key=lambda x: x[1])[0]

    # Find connected components of cells that are NOT barrier_color (4-connectivity)
    visited = [[False]*w for _ in range(h)]
    comp_id = [[-1]*w for _ in range(h)]
    comps = {}  # id -> dict with size, minr, minc, cells list optional
    drs = [(-1,0),(1,0),(0,-1),(0,1)]
    next_id = 0

    for r in range(h):
        for c in range(w):
            if grid[r][c] == barrier_color or visited[r][c]:
                continue
            # BFS for this component
            q = deque()
            q.append((r,c))
            visited[r][c] = True
            comp_id[r][c] = next_id
            size = 0
            minr = r
            minc = c
            while q:
                cr, cc = q.popleft()
                size += 1
                if cr < minr: minr = cr
                if cc < minc: minc = cc
                for dr, dc in drs:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] != barrier_color:
                        visited[nr][nc] = True
                        comp_id[nr][nc] = next_id
                        q.append((nr,nc))
            comps[next_id] = (size, minr, minc)
            next_id += 1

    # If there are no components (everything is barrier), return original
    if not comps:
        return [row[:] for row in grid]

    # Choose largest component; tie-break by smaller minr then smaller minc
    # We will maximize (size, -minr, -minc) so that larger size chosen; if tie choose smallest minr/minc
    best_id = max(comps.items(), key=lambda it: (it[1][0], -it[1][1], -it[1][2]))[0]

    # Build output: barrier cells unchanged; cells in best comp -> 4, others -> 3
    out = [[None]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == barrier_color:
                out[r][c] = grid[r][c]
            else:
                cid = comp_id[r][c]
                if cid == best_id:
                    out[r][c] = 4
                else:
                    out[r][c] = 3

    return out