def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    original = [row[:] for row in grid_lst]
    output = [row[:] for row in original]

    # Compute original red blocks for each row: list of (start, end) 0-indexed
    blocks = [[] for _ in range(rows)]
    for r in range(rows):
        i = 0
        while i < cols:
            if original[r][i] == 2:
                start = i
                while i < cols and original[r][i] == 2:
                    i += 1
                end = i - 1
                blocks[r].append((start, end))
            else:
                i += 1

    # Horizontal fills
    for r in range(rows):
        blk = blocks[r]
        nblk = len(blk)
        if nblk == 1:
            s, e = blk[0]
            if e - s + 1 >= 2:
                for c in range(s):
                    output[r][c] = 1
        elif nblk > 1:
            for k in range(1, nblk):
                prev_e = blk[k - 1][1]
                curr_s = blk[k][0]
                for c in range(prev_e + 1, curr_s):
                    output[r][c] = 1

    # Find stretches for support
    i = 0
    while i < rows:
        if len(blocks[i]) <= 1:
            i += 1
            continue
        r_start = i
        while i < rows and len(blocks[i]) > 1:
            i += 1
        r_end = i - 1
        h = r_end - r_start + 1
        # Take blocks from r_start
        blk = blocks[r_start]
        if len(blk) != 2:
            continue
        left_s, left_e = blk[0]
        right_s, right_e = blk[1]
        gap_l = left_e + 1
        gap_r = right_s - 1
        if gap_l > gap_r:
            continue
        g_width = gap_r - gap_l + 1
        n_width = g_width // 2
        if n_width == 0:
            continue
        n_start = gap_r - n_width + 1
        n_end = gap_r
        support_r_start = r_end + 1
        base_r_start = support_r_start + h
        # Narrow
        for ii in range(h):
            rr = support_r_start + ii
            if 0 <= rr < rows:
                c_start = max(0, n_start)
                c_end = min(cols - 1, n_end)
                for c in range(c_start, c_end + 1):
                    output[rr][c] = 1
        # Base
        b_start_c = gap_l
        b_end_c = right_e
        for ii in range(h):
            rr = base_r_start + ii
            if 0 <= rr < rows:
                c_start = max(0, b_start_c)
                c_end = min(cols - 1, b_end_c)
                for c in range(c_start, c_end + 1):
                    output[rr][c] = 1

    # Vertical fills based on original
    for c in range(cols):
        red_rs = [r for r in range(rows) if original[r][c] == 2]
        for j in range(len(red_rs) - 1):
            p = red_rs[j]
            q = red_rs[j + 1]
            if q == p + 3:
                g1 = p + 1
                g2 = p + 2
                # 3-wide at g1
                c_start = max(0, c - 1)
                c_end = min(cols - 1, c + 1)
                if 0 <= g1 < rows:
                    for cc in range(c_start, c_end + 1):
                        output[g1][cc] = 1
                # 1-wide at g2
                if 0 <= g2 < rows:
                    output[g2][c] = 1

    return output