def get_dir(dr, dc):
    deltas = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    for i, (ddr, ddc) in enumerate(deltas):
        if ddr == dr and ddc == dc:
            return i
    raise ValueError("Invalid direction")

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])
    ones = set()
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 1:
                ones.add((r, c))
    if not ones:
        return [[]]
    # Find start: min r, min c
    min_r = min(r for r, _ in ones)
    candidates = [(r, c) for r, c in ones if r == min_r]
    start = min(candidates, key=lambda p: p[1])
    # Deltas for 8-connected
    deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    def get_neighbors(p):
        r, c = p
        nb = []
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in ones:
                nb.append((nr, nc))
        return nb
    neighs = get_neighbors(start)
    if len(neighs) != 2:
        return [[]]  # invalid, but assume valid
    # Pick initial next: the one with smaller dir
    dir0 = get_dir(neighs[0][0] - start[0], neighs[0][1] - start[1])
    dir1 = get_dir(neighs[1][0] - start[0], neighs[1][1] - start[1])
    initial_next = neighs[0] if dir0 < dir1 else neighs[1]
    # Trace path
    path = [start, initial_next]
    current = initial_next
    prev = start
    while current != start:
        neighs_curr = get_neighbors(current)
        next_candidates = [nb for nb in neighs_curr if nb != prev]
        if len(next_candidates) != 1:
            return [[]]  # invalid
        next_p = next_candidates[0]
        path.append(next_p)
        prev = current
        current = next_p
    path = path[:-1]  # remove duplicate start
    n = len(path)
    if n == 0:
        return [[]]
    # Compute dir_list
    dir_list = []
    for i in range(n):
        p1_r, p1_c = path[i]
        p2_r, p2_c = path[(i + 1) % n]
        dr = p2_r - p1_r
        dc = p2_c - p1_c
        d = get_dir(dr, dc)
        dir_list.append(d)
    # Find runs: list of (dir, len)
    runs = []
    if n > 0:
        current_dir = dir_list[0]
        current_len = 1
        for i in range(1, n):
            if dir_list[i] == current_dir:
                current_len += 1
            else:
                runs.append((current_dir, current_len))
                current_dir = dir_list[i]
                current_len = 1
        runs.append((current_dir, current_len))
    # Now, n_sides = len(runs)
    n_sides = len(runs)
    # Merge short len=1 adj runs
    i = 0
    while i < n_sides - 1:
        j = i + 1
        diff = abs(runs[i][0] - runs[j][0])
        min_diff = min(diff, 8 - diff)
        if runs[i][1] == 1 and min_diff == 1:
            # merge: remove i, add len to j? but since 1, just remove i
            del runs[i]
            n_sides -= 1
            # do not i+=1, check again from i
        else:
            i += 1
    # Check wrap around
    if n_sides > 1:
        diff = abs(runs[-1][0] - runs[0][0])
        min_diff = min(diff, 8 - diff)
        if runs[-1][1] == 1 and min_diff == 1:
            n_sides -= 1
    return [[7] * n_sides]