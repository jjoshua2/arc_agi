def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    h = len(grid_lst)
    if h == 0:
        return []
    w = len(grid_lst[0])
    input_grid = grid_lst
    output = [row[:] for row in grid_lst]
    movers = []
    target_set = set()
    for r in range(h):
        for c in range(w):
            if input_grid[r][c] != 1:
                continue
            # horizontal
            has_h = False
            min_dh = float('inf')
            closest_ch = -1
            for cc in range(w):
                if input_grid[r][cc] == 2:
                    has_h = True
                    d = abs(c - cc)
                    if d < min_dh or (d == min_dh and (closest_ch == -1 or cc < closest_ch)):
                        min_dh = d
                        closest_ch = cc
            # vertical
            has_v = False
            min_dv = float('inf')
            closest_rv = -1
            for rr in range(h):
                if input_grid[rr][c] == 2:
                    has_v = True
                    d = abs(r - rr)
                    if d < min_dv or (d == min_dv and (closest_rv == -1 or rr < closest_rv)):
                        min_dv = d
                        closest_rv = rr
            # check if already adjacent
            already_adj_h = has_h and min_dh == 1
            already_adj_v = has_v and min_dv == 1
            if already_adj_h or already_adj_v or (not has_h and not has_v):
                continue
            # need to move
            move_dir = None
            if not has_h:
                move_dir = 'v'
            elif not has_v:
                move_dir = 'h'
            else:
                if min_dh < min_dv:
                    move_dir = 'h'
                elif min_dv < min_dh:
                    move_dir = 'v'
                else:
                    move_dir = 'h'  # arbitrary for tie
            # compute target
            if move_dir == 'h':
                delta = 1 if c > closest_ch else -1
                tr = r
                tc = closest_ch + delta
            else:
                delta = 1 if r > closest_rv else -1
                tr = closest_rv + delta
                tc = c
            # assume in bounds and valid
            movers.append((r, c))
            target_set.add((tr, tc))
    # apply changes
    for mr, mc in movers:
        output[mr][mc] = 0
    for tr, tc in target_set:
        if 0 <= tr < h and 0 <= tc < w:
            output[tr][tc] = 1
    return output