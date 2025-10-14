def get_neighbors(r, c, rows, cols):
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    ns = []
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            ns.append((nr, nc))
    return ns

def transform(grid_lst):
    if not grid_lst or not grid_lst[0]:
        return []
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    grid = [row[:] for row in grid_lst]
    # Find all targets: 0's adjacent to 2's
    targets = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                for nr, nc in get_neighbors(r, c, rows, cols):
                    if grid[nr][nc] == 0:
                        targets.add((nr, nc))
    # Find all blues
    blues = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]
    # For each blue, move to closest target in row or col
    for br, bc in blues:
        candidates = []
        # Same row targets
        for tr, tc in targets:
            if tr == br and tc != bc:
                dist = abs(tc - bc)
                candidates.append((dist, tr, tc))
        # Same col targets
        for tr, tc in targets:
            if tc == bc and tr != br:
                dist = abs(tr - br)
                candidates.append((dist, tr, tc))
        if not candidates:
            continue
        # Sort to get min dist, then smallest tr, then smallest tc
        candidates.sort()
        min_dist, tr, tc = candidates[0]
        # Move
        grid[tr][tc] = 1
        grid[br][bc] = 0
    return grid