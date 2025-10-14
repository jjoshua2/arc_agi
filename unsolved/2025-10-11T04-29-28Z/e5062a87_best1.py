def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])
    # find red positions
    red_positions = [(i, j) for i in range(h) for j in range(w) if grid[i][j] == 2]
    if not red_positions:
        return [row[:] for row in grid]
    min_r = min(i for i, j in red_positions)
    max_r = max(i for i, j in red_positions)
    min_c = min(j for i, j in red_positions)
    max_c = max(j for i, j in red_positions)
    # find accents
    accents = [(i, j, grid[i][j]) for i in range(h) for j in range(w) if grid[i][j] != 0 and grid[i][j] != 2]
    out = [row[:] for row in grid]
    # upper
    upper_accent_cols = {j for i, j, col in accents if i < min_r}
    fill_cols_upper = {j for j in range(min_c, max_c + 1) if j not in upper_accent_cols}
    for i in range(min_r):
        for j in range(w):
            if out[i][j] == 0 and j in fill_cols_upper:
                out[i][j] = 2
    # lower
    lower_accent_cols = {j for i, j, col in accents if i > max_r}
    fill_cols_lower = {j for j in range(min_c, max_c + 1) if j not in lower_accent_cols}
    for i in range(max_r + 1, h):
        for j in range(w):
            if out[i][j] == 0 and j in fill_cols_lower:
                out[i][j] = 2
    # main rows
    for i in range(min_r, max_r + 1):
        row_accents = [(j, grid[i][j]) for j in range(w) if grid[i][j] != 0 and grid[i][j] != 2]
        if not row_accents:
            # full fill
            for j in range(w):
                if out[i][j] == 0:
                    out[i][j] = 2
            continue
        # has accent
        has_left = any(j < min_c for j, col in row_accents)
        has_right = any(j > max_c for j, col in row_accents)
        if has_right:
            for j in range(max_c + 1):
                if out[i][j] == 0:
                    out[i][j] = 2
        if has_left:
            for j in range(min_c, w):
                if out[i][j] == 0:
                    out[i][j] = 2
    return out