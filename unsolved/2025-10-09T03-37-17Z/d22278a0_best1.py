def fill_line(w, r, color, is_normal):
    leading = r + 1
    line = [0] * w
    if leading >= w:
        if r % 2 == 0:
            for i in range(w):
                line[i] = color
        # else remains 0
    else:
        if r % 2 == 0:
            # even: leading C, then repeat 0 C
            for i in range(leading):
                line[i] = color
            pos = leading
            for j in range(w - leading):
                if j % 2 == 0:
                    line[pos] = 0
                else:
                    line[pos] = color
                pos += 1
        else:
            # odd: leading 0, then repeat C 0
            for i in range(leading):
                line[i] = 0
            pos = leading
            for j in range(w - leading):
                if j % 2 == 0:
                    line[pos] = color
                else:
                    line[pos] = 0
                pos += 1
    if not is_normal:
        line = line[::-1]
    return line

def transform(grid_lst):
    if not grid_lst or not grid_lst[0]:
        return []
    H = len(grid_lst)
    W = len(grid_lst[0])
    grid = [[0] * W for _ in range(H)]

    # Collect anchors
    top_anchors = [(c, grid_lst[0][c]) for c in range(W) if grid_lst[0][c] != 0]
    bottom_anchors = [(c, grid_lst[H-1][c]) for c in range(W) if grid_lst[H-1][c] != 0]
    top_anchors.sort(key=lambda x: x[0])
    bottom_anchors.sort(key=lambda x: x[0])

    half_h = H // 2
    upper_range = list(range(0, half_h))
    lower_range = list(range(H - half_h, H))

    # Upper blocks
    upper_blocks = []
    has_left_upper = False
    has_right_upper = False
    upper_left_color = None
    upper_right_color = None
    if top_anchors:
        if len(top_anchors) >= 2:
            left_color = top_anchors[0][1]
            right_color = top_anchors[-1][1]
            bw = W // 2
            upper_blocks.append((0, bw, left_color, True))  # normal
            rs = W - bw
            upper_blocks.append((rs, bw, right_color, False))  # mirrored
            has_left_upper = True
            has_right_upper = True
            upper_left_color = left_color
            upper_right_color = right_color
        else:
            c, color = top_anchors[0]
            if c <= W // 2:
                has_left_upper = True
                upper_left_color = color
                if len(bottom_anchors) >= 2:
                    other_opp_w = W // 2
                    sw = W - other_opp_w + (W % 2)
                    bs = 0
                    d = True  # normal
                else:
                    sw = W
                    bs = 0
                    d = True
                upper_blocks.append((bs, sw, color, d))
            else:
                has_right_upper = True
                upper_right_color = color
                if len(bottom_anchors) >= 2:
                    other_opp_w = W // 2
                    sw = W - other_opp_w + (W % 2)
                    bs = W - sw
                    d = False  # mirrored
                else:
                    sw = W
                    bs = 0
                    d = False
                upper_blocks.append((bs, sw, color, d))

    # Lower blocks
    lower_blocks = []
    has_left_lower = False
    has_right_lower = False
    lower_left_color = None
    lower_right_color = None
    if bottom_anchors:
        if len(bottom_anchors) >= 2:
            left_color = bottom_anchors[0][1]
            right_color = bottom_anchors[-1][1]
            bw = W // 2
            lower_blocks.append((0, bw, left_color, True))  # normal
            rs = W - bw
            lower_blocks.append((rs, bw, right_color, False))  # mirrored
            has_left_lower = True
            has_right_lower = True
            lower_left_color = left_color
            lower_right_color = right_color
        else:
            c, color = bottom_anchors[0]
            if c <= W // 2:
                has_left_lower = True
                lower_left_color = color
                if len(top_anchors) >= 2:
                    other_opp_w = W // 2
                    sw = W - other_opp_w + (W % 2)
                    bs = 0
                    d = True
                else:
                    sw = W
                    bs = 0
                    d = True
                lower_blocks.append((bs, sw, color, d))
            else:
                has_right_lower = True
                lower_right_color = color
                if len(top_anchors) >= 2:
                    other_opp_w = W // 2
                    sw = W - other_opp_w + (W % 2)
                    bs = W - sw
                    d = False
                else:
                    sw = W
                    bs = 0
                    d = False
                lower_blocks.append((bs, sw, color, d))

    # Fill upper
    for bs, bw, colr, is_normal in upper_blocks:
        for row in upper_range:
            rr = row
            line = fill_line(bw, rr, colr, is_normal)
            for i in range(bw):
                grid[row][bs + i] = line[i]

    # Fill lower
    for bs, bw, colr, is_normal in lower_blocks:
        for row in lower_range:
            rr = H - 1 - row
            line = fill_line(bw, rr, colr, is_normal)
            for i in range(bw):
                grid[row][bs + i] = line[i]

    # Middle
    if H % 2 == 1:
        mid = half_h
        for c in range(W):
            grid[mid][c] = 0

    # Continuations
    if len(bottom_anchors) == 0:
        for bs, bw, colr, is_normal in upper_blocks:
            for row in lower_range:
                rr = row
                line = fill_line(bw, rr, colr, is_normal)
                for i in range(bw):
                    grid[row][bs + i] = line[i]
    if len(top_anchors) == 0:
        for bs, bw, colr, is_normal in lower_blocks:
            for row in upper_range:
                rr = H - 1 - row
                line = fill_line(bw, rr, colr, is_normal)
                for i in range(bw):
                    grid[row][bs + i] = line[i]

    # Spills
    if len(top_anchors) > 0 and len(bottom_anchors) > 0:
        # upper to lower right spill
        if has_right_upper and not has_right_lower:
            uc = upper_right_color
            for g in lower_range:
                if g % 2 == 0:
                    length = H - g - 1
                    if length > 0:
                        sc = W - length
                        for i in range(length):
                            grid[g][sc + i] = uc
                        ns = sc - 2
                        for i in range(2):
                            cc = ns + i
                            if 0 <= cc < W:
                                grid[g][cc] = 0
        # upper to lower left spill
        if has_left_upper and not has_left_lower:
            uc = upper_left_color
            for g in lower_range:
                if g % 2 == 0:
                    length = H - g - 1
                    if length > 0:
                        sc = 0
                        for i in range(length):
                            grid[g][sc + i] = uc
                        ns = sc + length
                        for i in range(2):
                            cc = ns + i
                            if cc < W:
                                grid[g][cc] = 0
        # lower to upper left spill
        if has_left_lower and not has_left_upper:
            lc = lower_left_color
            for g in upper_range:
                if g % 2 == 1:
                    length = g + 1 if g == 0 else g  # adjust if needed, but g odd >=1
                    sc = 0
                    for i in range(length):
                        grid[g][sc + i] = lc
                    ns = sc + length
                    for i in range(2):
                        cc = ns + i
                        if cc < W:
                            grid[g][cc] = 0
        # lower to upper right spill
        if has_right_lower and not has_right_upper:
            lc = lower_right_color
            for g in upper_range:
                if g % 2 == 1:
                    length = g
                    sc = W - length
                    for i in range(length):
                        grid[g][sc + i] = lc
                    ns = sc - 2
                    for i in range(2):
                        cc = ns + i
                        if 0 <= cc < W:
                            grid[g][cc] = 0

    return grid