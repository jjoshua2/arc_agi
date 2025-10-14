def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return [row[:] for row in grid]
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]
    from collections import defaultdict
    color_count = defaultdict(int)
    for r in range(rows):
        for c in range(cols):
            color_count[grid[r][c]] += 1
    non_zero_colors = [k for k in color_count if k != 0 and color_count[k] > 1]  # main colors
    if len(non_zero_colors) < 2:
        return out
    B = grid[0][0]
    I = [c for c in non_zero_colors if c != B][0]
    # Find seeds
    seeds = []
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v not in (0, B, I):
                # determine region
                num_i = 0
                num_b = 0
                for drr, dcc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + drr, c + dcc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        vv = grid[nr][nc]
                        if vv == I:
                            num_i += 1
                        if vv == B:
                            num_b += 1
                is_inner = num_i >= num_b
                R = I if is_inner else B
                other = B if is_inner else I
                # precompute narrow for this region
                max_span = 0
                for rr in range(rows):
                    current_span = 0
                    for cc in range(cols):
                        if grid[rr][cc] == R:
                            current_span += 1
                            max_span = max(max_span, current_span)
                        else:
                            current_span = 0
                threshold = max_span // 2 + 1
                narrow = set()
                last_narrow = -1
                for rr in range(rows):
                    current_span = 0
                    max_s = 0
                    for cc in range(cols):
                        if grid[rr][cc] == R:
                            current_span += 1
                            max_s = max(max_s, current_span)
                        else:
                            current_span = 0
                    if max_s <= threshold:
                        narrow.add(rr)
                        last_narrow = max(last_narrow, rr)
                seeds.append((r, c, v, is_inner, R, other, narrow, last_narrow))
    # Directions
    directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for sr, sc, S, is_inner, R, other, narrow, last_narrow in seeds:
        for dr, dc in directions:
            cr = sr
            cc = sc
            crossing = False
            while True:
                # compute next
                nr = cr + dr
                nc = cc + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    # out
                    if is_inner:
                        break
                    # for border, special up left out
                    if dr < 0 and dc < 0 and nc < 0:
                        # projected to row 1
                        steps = cr - 1
                        virtual_c = cc + steps * dc
                        if virtual_c < 0:
                            if 0 <= 1 < rows and grid[1][0] == R:
                                out[1][0] = S
                        break
                    break
                if grid[nr][nc] == R:
                    out[nr][nc] = S
                    cr = nr
                    cc = nc
                    crossing = False  # reset if normal move
                    continue
                # hit non R
                if not is_inner and grid[nr][nc] == other:
                    # transparent for border
                    cr = nr
                    cc = nc
                    # no paint
                    continue
                # wall, try reflect
                reflected = False
                new_dr = dr
                new_dc = dc
                vertical_hit = 0 <= nr < rows and 0 <= cc < cols and grid[nr][cc] == R
                horizontal_hit = 0 <= cr < rows and 0 <= nc < cols and grid[cr][nc] == R
                if vertical_hit:
                    new_dc = -dc
                    reflected = True
                elif horizontal_hit:
                    new_dr = -dr
                    reflected = True
                if not reflected:
                    break
                # reflected
                # new next
                nnr = cr + new_dr
                nnc = cc + new_dc
                if not (0 <= nnr < rows and 0 <= nnc < cols) or grid[nnr][nnc] != R:
                    break
                if crossing:
                    # second hit
                    # paint current
                    out[cr][cc] = S
                    # check if propagate
                    if dr > 0 and nr in narrow and nr == last_narrow:
                        # reflect again
                        reflected2 = False
                        new_dr2 = new_dr
                        new_dc2 = new_dc
                        vertical_hit2 = 0 <= nr < rows and 0 <= cc < cols and grid[nr][cc] == R
                        horizontal_hit2 = 0 <= cr < rows and 0 <= nc < cols and grid[cr][nc] == R
                        if vertical_hit2:
                            new_dc2 = -new_dc
                            reflected2 = True
                        elif horizontal_hit2:
                            new_dr2 = -new_dr
                            reflected2 = True
                        if reflected2:
                            crossing = False
                            dr = new_dr2
                            dc = new_dc2
                            # move to new next
                            nnr = cr + dr
                            nnc = cc + dc
                            if 0 <= nnr < rows and 0 <= nnc < cols and grid[nnr][nnc] == R:
                                out[nnr][nnc] = S
                                cr = nnr
                                cc = nnc
                            else:
                                break
                            continue
                    # not propagate
                    crossing = False
                    break
                else:
                    # first reflection, set crossing if inner and dr >0 and current row in narrow
                    if is_inner and dr > 0 and cr in narrow:
                        crossing = True
                    # move to new next without paint if crossing
                    cr = nnr
                    cc = nnc
                    if crossing:
                        # no paint
                        pass
                    else:
                        out[cr][cc] = S
    return out