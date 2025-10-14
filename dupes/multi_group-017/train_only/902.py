def check_vertical_painted_bottom(grid):
    rows = len(grid)
    cols = len(grid[0])
    n_half = rows // 2
    mismatches = []
    for r in range(n_half):
        b = rows - 1 - r
        for c in range(cols):
            if grid[r][c] != grid[b][c]:
                mismatches.append((b, c))
    if not mismatches:
        return None
    colors = set(grid[p[0]][p[1]] for p in mismatches)
    if len(colors) != 1:
        return None
    min_b = min(p[0] for p in mismatches)
    max_b = max(p[0] for p in mismatches)
    min_c = min(p[1] for p in mismatches)
    max_c = max(p[1] for p in mismatches)
    h = max_b - min_b + 1
    w = max_c - min_c + 1
    if len(mismatches) != h * w:
        return None
    output = []
    for i in range(h):
        b = min_b + i
        top_r = rows - 1 - b
        row = [grid[top_r][min_c + j] for j in range(w)]
        output.append(row)
    return output

def check_vertical_painted_top(grid):
    rows = len(grid)
    cols = len(grid[0])
    n_half = rows // 2
    mismatches = []
    for r in range(n_half):
        b = rows - 1 - r
        for c in range(cols):
            if grid[r][c] != grid[b][c]:
                mismatches.append((r, c))
    if not mismatches:
        return None
    colors = set(grid[p[0]][p[1]] for p in mismatches)
    if len(colors) != 1:
        return None
    min_r = min(p[0] for p in mismatches)
    max_r = max(p[0] for p in mismatches)
    min_c = min(p[1] for p in mismatches)
    max_c = max(p[1] for p in mismatches)
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    if len(mismatches) != h * w:
        return None
    output = []
    for i in range(h):
        r = min_r + i
        bottom_r = rows - 1 - r
        row = [grid[bottom_r][min_c + j] for j in range(w)]
        output.append(row)
    return output

def check_horizontal_painted_right(grid):
    rows = len(grid)
    cols = len(grid[0])
    n_half = cols // 2
    mismatches = []
    for cc in range(n_half):
        right_c = cols - 1 - cc
        for r in range(rows):
            if grid[r][cc] != grid[r][right_c]:
                mismatches.append((r, right_c))
    if not mismatches:
        return None
    colors = set(grid[p[0]][p[1]] for p in mismatches)
    if len(colors) != 1:
        return None
    min_r = min(p[0] for p in mismatches)
    max_r = max(p[0] for p in mismatches)
    min_c = min(p[1] for p in mismatches)
    max_c = max(p[1] for p in mismatches)
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    if len(mismatches) != h * w:
        return None
    left_min_c = cols - 1 - max_c
    output = []
    for i in range(h):
        r = min_r + i
        row = [grid[r][left_min_c + j] for j in range(w)]
        row = row[::-1]  # mirror reverse
        output.append(row)
    return output

def check_horizontal_painted_left(grid):
    rows = len(grid)
    cols = len(grid[0])
    n_half = cols // 2
    mismatches = []
    for cc in range(n_half):
        right_c = cols - 1 - cc
        for r in range(rows):
            if grid[r][cc] != grid[r][right_c]:
                mismatches.append((r, cc))
    if not mismatches:
        return None
    colors = set(grid[p[0]][p[1]] for p in mismatches)
    if len(colors) != 1:
        return None
    min_r = min(p[0] for p in mismatches)
    max_r = max(p[0] for p in mismatches)
    min_c = min(p[1] for p in mismatches)
    max_c = max(p[1] for p in mismatches)
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    if len(mismatches) != h * w:
        return None
    right_min_c = cols - 1 - max_c
    output = []
    for i in range(h):
        r = min_r + i
        row = [grid[r][right_min_c + j] for j in range(w)]
        row = row[::-1]  # mirror reverse
        output.append(row)
    return output

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = [row[:] for row in grid_lst]  # deep copy if needed, but since read-only
    candidates = []
    v_bottom = check_vertical_painted_bottom(grid)
    if v_bottom is not None:
        candidates.append(v_bottom)
    v_top = check_vertical_painted_top(grid)
    if v_top is not None:
        candidates.append(v_top)
    h_right = check_horizontal_painted_right(grid)
    if h_right is not None:
        candidates.append(h_right)
    h_left = check_horizontal_painted_left(grid)
    if h_left is not None:
        candidates.append(h_left)
    if not candidates:
        return []
    # Pick the one with smallest area
    min_area = min(len(c) * len(c[0]) for c in candidates)
    for cand in candidates:
        if len(cand) * len(cand[0]) == min_area:
            return cand
    # fallback
    return candidates[0]