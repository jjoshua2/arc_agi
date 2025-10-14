def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = [row[:] for row in grid_lst]
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]

    # Filling gaps
    for r in range(rows):
        red_cols = [c for c in range(cols) if grid[r][c] == 2]
        if len(red_cols) >= 2:
            min_c = min(red_cols)
            max_c = max(red_cols)
            for c in range(min_c, max_c + 1):
                if grid[r][c] == 5:
                    output[r][c] = 8

    for c in range(cols):
        red_rows = [r for r in range(rows) if grid[r][c] == 2]
        if len(red_rows) >= 2:
            min_r = min(red_rows)
            max_r = max(red_rows)
            for rr in range(min_r, max_r + 1):
                if grid[rr][c] == 5:
                    output[rr][c] = 8

    # Define bars based on original red positions
    h_bars = []
    for r in range(rows):
        red_cols = [c for c in range(cols) if grid[r][c] == 2]
        if len(red_cols) >= 2:
            h_bars.append((r, min(red_cols), max(red_cols)))

    v_bars = []
    for c in range(cols):
        red_rows = [r for r in range(rows) if grid[r][c] == 2]
        if len(red_rows) >= 2:
            v_bars.append((c, min(red_rows), max(red_rows)))

    # Extensions for symmetry at intersections
    for r, h_left, h_right in h_bars:
        for c, v_top, v_bottom in v_bars:
            if h_left <= c <= h_right and v_top <= r <= v_bottom:
                up_len = r - v_top
                down_len = v_bottom - r
                left_len = c - h_left
                right_len = h_right - c

                max_ud = max(up_len, down_len)
                if up_len < max_ud:
                    ext = max_ud - up_len
                    for k in range(1, ext + 1):
                        nr = v_top - k
                        if 0 <= nr < rows and grid[nr][c] == 5:
                            output[nr][c] = 8
                if down_len < max_ud:
                    ext = max_ud - down_len
                    for k in range(1, ext + 1):
                        nr = v_bottom + k
                        if 0 <= nr < rows and grid[nr][c] == 5:
                            output[nr][c] = 8

                max_lr = max(left_len, right_len)
                if left_len < max_lr:
                    ext = max_lr - left_len
                    for k in range(1, ext + 1):
                        nc = h_left - k
                        if 0 <= nc < cols and grid[r][nc] == 5:
                            output[r][nc] = 8
                if right_len < max_lr:
                    ext = max_lr - right_len
                    for k in range(1, ext + 1):
                        nc = h_right + k
                        if 0 <= nc < cols and grid[r][nc] == 5:
                            output[r][nc] = 8

    return output