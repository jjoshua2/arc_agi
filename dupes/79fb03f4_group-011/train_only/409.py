def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return grid_lst
    grid = [row[:] for row in grid_lst]
    h = len(grid)
    w = len(grid[0])
    seed_rows = [r for r in range(h) if grid[r][0] == 1]
    for r in sorted(seed_rows):
        horizontal_propagate(grid, r, h, w)
    return grid

def horizontal_propagate(grid, r, h, w):
    j = 0
    while j < w - 1:
        next_val = grid[r][j + 1]
        if next_val == 0:
            grid[r][j + 1] = 1
            j += 1
            continue
        elif next_val == 1:
            j += 1
            continue
        elif next_val in (5, 8):
            c = j + 1
            can_bypass = False
            right_filled_any = False
            # up
            if r > 0:
                s = r - 1
                l_c = c - 1 if c > 0 else -1
                cen_c = c
                ri_c = c + 1 if c + 1 < w else -1
                left_ok = (l_c < 0 or grid[s][l_c] in (0, 1))
                center_ok = grid[s][cen_c] in (0, 1)
                if left_ok and center_ok:
                    can_bypass = True
                    right_ok = (ri_c < 0 or grid[s][ri_c] in (0, 1))
                    if right_ok:
                        right_filled_any = True
                    # fill 0's
                    if grid[s][cen_c] == 0:
                        grid[s][cen_c] = 1
                    if l_c >= 0 and grid[s][l_c] == 0:
                        grid[s][l_c] = 1
                    if ri_c >= 0 and grid[s][ri_c] == 0:
                        grid[s][ri_c] = 1
                    # bypass if right was obstacle
                    if ri_c >= 0 and grid[s][ri_c] in (5, 8):
                        bypass_propagate(grid, s, ri_c, h, w)
            # down
            if r < h - 1:
                s = r + 1
                l_c = c - 1 if c > 0 else -1
                cen_c = c
                ri_c = c + 1 if c + 1 < w else -1
                left_ok = (l_c < 0 or grid[s][l_c] in (0, 1))
                center_ok = grid[s][cen_c] in (0, 1)
                if left_ok and center_ok:
                    can_bypass = True
                    right_ok = (ri_c < 0 or grid[s][ri_c] in (0, 1))
                    if right_ok:
                        right_filled_any = True
                    # fill
                    if grid[s][cen_c] == 0:
                        grid[s][cen_c] = 1
                    if l_c >= 0 and grid[s][l_c] == 0:
                        grid[s][l_c] = 1
                    if ri_c >= 0 and grid[s][ri_c] == 0:
                        grid[s][ri_c] = 1
                    if ri_c >= 0 and grid[s][ri_c] in (5, 8):
                        bypass_propagate(grid, s, ri_c, h, w)
            if can_bypass and right_filled_any:
                j = c
            else:
                break
        else:
            break

def bypass_propagate(grid, s, obs_c, h, w):
    bypass_col = obs_c + 1
    if bypass_col < w and grid[s][bypass_col] == 0:
        grid[s][bypass_col] = 1
    # vertical branch from this obstacle
    vertical_branch(grid, s, obs_c, h, w)

def vertical_branch(grid, r, c, h, w):
    # up
    if r > 0:
        s = r - 1
        l_c = c - 1 if c > 0 else -1
        cen_c = c
        ri_c = c + 1 if c + 1 < w else -1
        left_ok = (l_c < 0 or grid[s][l_c] in (0, 1))
        center_ok = grid[s][cen_c] in (0, 1)
        if left_ok and center_ok:
            if grid[s][cen_c] == 0:
                grid[s][cen_c] = 1
            if l_c >= 0 and grid[s][l_c] == 0:
                grid[s][l_c] = 1
            if ri_c >= 0 and grid[s][ri_c] == 0:
                grid[s][ri_c] = 1
            if ri_c >= 0 and grid[s][ri_c] in (5, 8):
                bypass_propagate(grid, s, ri_c, h, w)
    # down
    if r < h - 1:
        s = r + 1
        l_c = c - 1 if c > 0 else -1
        cen_c = c
        ri_c = c + 1 if c + 1 < w else -1
        left_ok = (l_c < 0 or grid[s][l_c] in (0, 1))
        center_ok = grid[s][cen_c] in (0, 1)
        if left_ok and center_ok:
            if grid[s][cen_c] == 0:
                grid[s][cen_c] = 1
            if l_c >= 0 and grid[s][l_c] == 0:
                grid[s][l_c] = 1
            if ri_c >= 0 and grid[s][ri_c] == 0:
                grid[s][ri_c] = 1
            if ri_c >= 0 and grid[s][ri_c] in (5, 8):
                bypass_propagate(grid, s, ri_c, h, w)