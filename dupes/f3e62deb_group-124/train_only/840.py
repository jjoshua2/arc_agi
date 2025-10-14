def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    rows = len(grid)
    cols = len(grid[0])
    non_zero_pos = []
    C = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                if C is None:
                    C = grid[r][c]
                non_zero_pos.append((r, c))
    if not non_zero_pos:
        return [row[:] for row in grid]
    min_r = min(p[0] for p in non_zero_pos)
    min_c = min(p[1] for p in non_zero_pos)
    if C == 8:
        delta_r = 0
        delta_c = 7 - min_c
    elif C == 4:
        delta_r = 7 - min_r
        delta_c = 0
    elif C == 6:
        delta_r = 0 - min_r
        delta_c = 0
    else:
        return [row[:] for row in grid]
    out = [row[:] for row in grid]
    for r, c in non_zero_pos:
        out[r][c] = 0
    for r, c in non_zero_pos:
        new_r = r + delta_r
        new_c = c + delta_c
        out[new_r][new_c] = C
    return out