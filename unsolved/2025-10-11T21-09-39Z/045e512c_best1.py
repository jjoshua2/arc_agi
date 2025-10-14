def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    grid = [row[:] for row in grid_lst]
    output = [row[:] for row in grid]
    # Find template: 3x3 region with uniform T !=0, count_t in [5,8)
    template = None
    for start_r in range(rows - 2):
        for start_c in range(cols - 2):
            sub = [[grid[start_r + i][start_c + j] for j in range(3)] for i in range(3)]
            non_zero_vals = [sub[i][j] for i in range(3) for j in range(3) if sub[i][j] != 0]
            if non_zero_vals:
                ts = set(non_zero_vals)
                if len(ts) == 1:
                    t = list(ts)[0]
                    count_t = sum(1 for i in range(3) for j in range(3) if sub[i][j] == t)
                    if 5 <= count_t < 9:
                        mask = [[1 if sub[i][j] == t else 0 for j in range(3)] for i in range(3)]
                        template = (start_r, start_c, t, mask)
                        break
        if template:
            break
    if not template:
        return output
    min_r, min_c, t, mask = template
    h = 3
    w = 3
    period = 4
    # Collect unique activations per S
    from collections import defaultdict
    activated = defaultdict(set)  # S: set of (offset_r, offset_c, sign_dr, sign_dc)
    for sr in range(rows):
        for sc in range(cols):
            s = grid[sr][sc]
            if s == 0 or s == t:
                continue
            delta_r = sr - min_r
            delta_c = sc - min_c
            offset_r = (delta_r // period) * period
            offset_c = (delta_c // period) * period
            rel_i = sr - (min_r + offset_r)
            rel_j = sc - (min_c + offset_c)
            if 0 <= rel_i < h and 0 <= rel_j < w and mask[rel_i][rel_j] == 1:
                sign_dr = 0 if delta_r == 0 else (1 if delta_r > 0 else -1)
                sign_dc = 0 if delta_c == 0 else (1 if delta_c > 0 else -1)
                key = (offset_r, offset_c, sign_dr, sign_dc)
                activated[s].add(key)
    # Fill from activations
    for s, keys in activated.items():
        for offset_r, offset_c, sign_dr, sign_dc in keys:
            step_r = sign_dr * period
            step_c = sign_dc * period
            cur_or = offset_r
            cur_oc = offset_c
            while True:
                br = min_r + cur_or
                bc = min_c + cur_oc
                # Check if any part in bounds
                any_in = any(0 <= br + ii < rows for ii in range(h))
                if not any_in:
                    break
                for ii in range(h):
                    tr = br + ii
                    if 0 <= tr < rows:
                        for jj in range(w):
                            tc = bc + jj
                            if 0 <= tc < cols and mask[ii][jj] == 1:
                                output[tr][tc] = s
                # Next position
                next_or = cur_or + step_r
                next_oc = cur_oc + step_c
                if step_r == 0 and step_c == 0:
                    break
                # Check next any in bounds
                next_br = min_r + next_or
                next_any = any(0 <= next_br + ii < rows for ii in range(h))
                if not next_any:
                    break
                cur_or = next_or
                cur_oc = next_oc
    return output