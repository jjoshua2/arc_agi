import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return [[0, 0, 0] for _ in range(3)]
    grid = np.array(grid_lst)
    r, c = grid.shape
    first = grid[0]
    non_zero = first[first != 0]
    if len(non_zero) == 0:
        return [[0, 0, 0] for _ in range(3)]
    unique_non0 = np.unique(non_zero)
    if len(unique_non0) != 1:
        # Filled case: dedup top row, wrap into 3x3 padding with 0
        out_colors = []
        prev = -1
        for val in first:
            if val != prev:
                out_colors.append(val)
                prev = val
        out = [[0, 0, 0] for _ in range(3)]
        idx = 0
        for i in range(3):
            for j in range(3):
                if idx < len(out_colors):
                    out[i][j] = out_colors[idx]
                    idx += 1
        return out
    line_color = unique_non0[0]
    v_mask = first == line_color
    v_cols = np.where(v_mask)[0].tolist()
    N = len(v_cols)
    if N < 2:
        return [[0, 0, 0] for _ in range(3)]
    spacing = v_cols[1] - v_cols[0]
    first_thick = spacing - 1
    thick_rows = []
    tr = first_thick
    while tr < r:
        thick_rows.append(tr)
        tr += spacing
    num_thick = len(thick_rows)
    if num_thick < 2:
        return [[0, 0, 0] for _ in range(3)]
    special_per_thick = []
    for ti in range(num_thick):
        tr = thick_rows[ti]
        s = {}
        for vi in range(N):
            vc = v_cols[vi]
            col = grid[tr, vc]
            if col != line_color and col != 0:
                if col not in s:
                    s[col] = []
                s[col].append(vi + 1)  # 1-based V index
        special_per_thick.append(s)
    contributions = [None] * (num_thick - 1)
    for p in range(num_thick - 1):
        s1 = special_per_thick[p]
        s2 = special_per_thick[p + 1]
        common = {}
        all_colors = set(s1.keys()) | set(s2.keys())
        for colr in all_colors:
            pos1 = s1.get(colr, [])
            pos2 = s2.get(colr, [])
            com_pos_set = set(pos1) & set(pos2)
            com_pos = [v for v in pos1 if v in com_pos_set]
            com_pos = sorted(set(com_pos))  # unique sorted
            segs = []
            if com_pos:
                curr = [com_pos[0]]
                for v in com_pos[1:]:
                    if v == curr[-1] + 1:
                        curr.append(v)
                    else:
                        if len(curr) >= 2:
                            segs.append((curr[0], curr[-1]))
                        curr = [v]
                if len(curr) >= 2:
                    segs.append((curr[0], curr[-1]))
            if segs:
                common[colr] = segs
        if common:
            contributions[p] = common
    output = [[0] * 3 for _ in range(3)]
    current_row = 0
    first_contrib = True
    last_p = -1
    for p in range(num_thick - 1):
        has = contributions[p] is not None
        if has:
            last_p = p
            common = contributions[p]
            for colr, segs in common.items():
                for a, b in segs:
                    k = b - a + 1
                    if k == 2:
                        first_v = a
                        offset = N // 3 - 1
                        col_idx = first_v - offset
                        col_idx = max(0, min(2, col_idx))
                        output[current_row][col_idx] = colr
                    else:
                        num_fill = k - 1
                        for jj in range(num_fill):
                            if jj < 3:
                                output[current_row][jj] = colr
            first_contrib = False
            current_row += 1
        else:
            if not first_contrib:
                current_row += 1
    # Full span of second of last_p
    if last_p >= 0 and current_row < 3:
        second_ti = last_p + 1
        s = special_per_thick[second_ti]
        for colr, poss in s.items():
            poss = sorted(poss)
            segs = []
            if poss:
                curr = [poss[0]]
                for v in poss[1:]:
                    if v == curr[-1] + 1:
                        curr.append(v)
                    else:
                        if len(curr) >= 2:
                            segs.append((curr[0], curr[-1]))
                        curr = [v]
                if len(curr) >= 2:
                    segs.append((curr[0], curr[-1]))
            for a, b in segs:
                k = b - a + 1
                num_fill = k - 1
                for jj in range(num_fill):
                    if jj < 3:
                        output[current_row][jj] = colr
    return output