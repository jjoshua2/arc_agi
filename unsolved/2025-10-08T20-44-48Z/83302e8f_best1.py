def transform(input_grid):
    grid = [row[:] for row in input_grid]  # copy
    h = len(grid)
    if h == 0:
        return grid
    w = len(grid[0])
    # find non zero positions
    non_zeros = [(i, j) for i in range(h) for j in range(w) if grid[i][j] != 0]
    if not non_zeros:
        return grid
    min_r = min(i for i, _ in non_zeros)
    max_r = max(i for i, _ in non_zeros)
    min_c = min(j for _, j in non_zeros)
    max_c = max(j for _, j in non_zeros)
    # interior fill
    for i in range(min_r, max_r + 1):
        for j in range(min_c, max_c + 1):
            if grid[i][j] == 0:
                grid[i][j] = 4
    # now pyramid
    empty_a = min_r
    empty_b = h - 1 - max_r
    empty_l = min_c
    empty_r = w - 1 - max_c
    max_empty = max(empty_a, empty_b, empty_l, empty_r)
    if max_empty == 0:
        return grid
    # find direction
    if empty_a == max_empty:
        # open up
        l = empty_a
        c = (w - 1) // 2
        for p in range(l):
            r = p
            if p == 0:
                le = c
                ri = c
            else:
                le = p - 1
                ri = w - 1 - (p - 1)
            if le > ri:
                continue
            dist = ri - le
            if dist <= 2:
                for jj in range(le, ri + 1):
                    if 0 <= jj < w and grid[r][jj] == 0:
                        grid[r][jj] = 4
            else:
                for jj in [le, c, ri]:
                    if 0 <= jj < w and grid[r][jj] == 0:
                        grid[r][jj] = 4
    elif empty_b == max_empty:
        # open down
        l = empty_b
        for p in range(l):
            r = max_r + 1 + p
            inset = p + 1
            le = min_c + inset
            ri = max_c - inset
            if le > ri:
                continue
            # always fill between
            for jj in range(le, ri + 1):
                if 0 <= jj < w and grid[r][jj] == 0:
                    grid[r][jj] = 4
            if p == l - 1:
                for jj in [min_c, max_c]:
                    if 0 <= jj < w and grid[r][jj] == 0:
                        grid[r][jj] = 4
    elif empty_l == max_empty:
        # open left
        l = empty_l
        u = min_r
        intrusion = max(0, l - u)
        upper_length = l
        for step in range(upper_length):
            rr = step
            cc = step
            if 0 <= rr < h and 0 <= cc < w and grid[rr][cc] == 0:
                grid[rr][cc] = 4
        lower_length = intrusion
        if lower_length > 0:
            end_row = max_r - (lower_length - 1)
            end_col = min_c - 1
            for step in range(lower_length):
                rr = end_row + step
                cc = end_col - step
                if 0 <= rr < h and 0 <= cc < w and grid[rr][cc] == 0:
                    grid[rr][cc] = 4
        # central full left
        upper_end = upper_length - 1
        lower_start = end_row if lower_length > 0 else h
        for rr in range(upper_end + 1, lower_start):
            for cc in range(min_c):
                if 0 <= cc < w and grid[rr][cc] == 0:
                    grid[rr][cc] = 4
    else:
        # open right
        l = empty_r
        u = min_r
        intrusion = max(0, l - u)
        upper_length = l
        for step in range(upper_length):
            rr = step
            cc = w - 1 - step
            if 0 <= rr < h and 0 <= cc < w and grid[rr][cc] == 0:
                grid[rr][cc] = 4
        lower_length = intrusion
        if lower_length > 0:
            end_row = max_r - (lower_length - 1)
            end_col = max_c + 1
            for step in range(lower_length):
                rr = end_row + step
                cc = end_col + step
                if 0 <= rr < h and 0 <= cc < w and grid[rr][cc] == 0:
                    grid[rr][cc] = 4
        # central full right
        upper_end = upper_length - 1
        lower_start = end_row if lower_length > 0 else h
        for rr in range(upper_end + 1, lower_start):
            for cc in range(max_c + 1, w):
                if 0 <= cc < w and grid[rr][cc] == 0:
                    grid[rr][cc] = 4
    return grid