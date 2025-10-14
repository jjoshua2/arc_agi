import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    output = grid.copy()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def get_non0_count(r_pos, c_pos):
        count = 0
        for dr, dc in directions:
            nr, nc = r_pos + dr, c_pos + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != 0:
                count += 1
        return count

    def get_side_deg(r_pos, c_pos, is_horizontal, side_dir):
        # side_dir 0 for up/left, 1 for down/right
        if is_horizontal:
            if side_dir == 0:
                sr, sc = r_pos - 1, c_pos
            else:
                sr, sc = r_pos + 1, c_pos
        else:
            if side_dir == 0:
                sr, sc = r_pos, c_pos - 1
            else:
                sr, sc = r_pos, c_pos + 1
        count = 0
        for dr, dc in directions:
            nr, nc = sr + dr, sc + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != 0:
                count += 1
        return count

    # Horizontal runs
    for r in range(rows):
        c = 0
        while c < cols:
            if grid[r, c] == 0:
                c += 1
                continue
            start_c = c
            two_cols = []
            while c < cols and grid[r, c] != 0:
                if grid[r, c] == 2:
                    two_cols.append(c)
                c += 1
            end_c = c - 1
            if not two_cols:
                continue
            min_c = min(two_cols)
            max_c = max(two_cols)
            # Fill inner
            for cc in range(min_c, max_c + 1):
                if grid[r, cc] == 5:
                    output[r, cc] = 8
            # Left extension
            if min_c > start_c:
                ext_c = min_c - 1
                if grid[r, ext_c] == 5:
                    non0 = get_non0_count(r, ext_c)
                    if non0 >= 2:
                        has_up = r - 1 >= 0 and grid[r - 1, ext_c] != 0
                        has_down = r + 1 < rows and grid[r + 1, ext_c] != 0
                        side_ok = not has_up and not has_down
                        if has_up:
                            deg = get_side_deg(r, ext_c, True, 0)
                            if deg >= 3:
                                side_ok = True
                        if has_down:
                            deg = get_side_deg(r, ext_c, True, 1)
                            if deg >= 3:
                                side_ok = True
                        if side_ok:
                            output[r, ext_c] = 8
            # Right extension
            if max_c < end_c:
                ext_c = max_c + 1
                if grid[r, ext_c] == 5:
                    non0 = get_non0_count(r, ext_c)
                    if non0 >= 2:
                        has_up = r - 1 >= 0 and grid[r - 1, ext_c] != 0
                        has_down = r + 1 < rows and grid[r + 1, ext_c] != 0
                        side_ok = not has_up and not has_down
                        if has_up:
                            deg = get_side_deg(r, ext_c, True, 0)
                            if deg >= 3:
                                side_ok = True
                        if has_down:
                            deg = get_side_deg(r, ext_c, True, 1)
                            if deg >= 3:
                                side_ok = True
                        if side_ok:
                            output[r, ext_c] = 8

    # Vertical runs
    for c in range(cols):
        r = 0
        while r < rows:
            if grid[r, c] == 0:
                r += 1
                continue
            start_r = r
            two_rows = []
            while r < rows and grid[r, c] != 0:
                if grid[r, c] == 2:
                    two_rows.append(r)
                r += 1
            end_r = r - 1
            if not two_rows:
                continue
            min_r = min(two_rows)
            max_r = max(two_rows)
            # Fill inner
            for rr in range(min_r, max_r + 1):
                if grid[rr, c] == 5:
                    output[rr, c] = 8
            # Top extension
            if min_r > start_r:
                ext_r = min_r - 1
                if grid[ext_r, c] == 5:
                    non0 = get_non0_count(ext_r, c)
                    if non0 >= 2:
                        has_left = c - 1 >= 0 and grid[ext_r, c - 1] != 0
                        has_right = c + 1 < cols and grid[ext_r, c + 1] != 0
                        side_ok = not has_left and not has_right
                        if has_left:
                            deg = get_side_deg(ext_r, c, False, 0)
                            if deg >= 3:
                                side_ok = True
                        if has_right:
                            deg = get_side_deg(ext_r, c, False, 1)
                            if deg >= 3:
                                side_ok = True
                        if side_ok:
                            output[ext_r, c] = 8
            # Bottom extension
            if max_r < end_r:
                ext_r = max_r + 1
                if grid[ext_r, c] == 5:
                    non0 = get_non0_count(ext_r, c)
                    if non0 >= 2:
                        has_left = c - 1 >= 0 and grid[ext_r, c - 1] != 0
                        has_right = c + 1 < cols and grid[ext_r, c + 1] != 0
                        side_ok = not has_left and not has_right
                        if has_left:
                            deg = get_side_deg(ext_r, c, False, 0)
                            if deg >= 3:
                                side_ok = True
                        if has_right:
                            deg = get_side_deg(ext_r, c, False, 1)
                            if deg >= 3:
                                side_ok = True
                        if side_ok:
                            output[ext_r, c] = 8

    return output.tolist()