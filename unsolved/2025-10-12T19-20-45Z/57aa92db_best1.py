import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    new_grid = grid.copy().astype(int)

    # Step 1: 8 expansions
    visited = np.zeros((rows, cols), dtype=bool)
    rects = []
    for r in range(rows):
        c = 0
        while c < cols:
            if visited[r, c] or grid[r, c] == 0:
                c += 1
                continue
            color = grid[r, c]
            # extend right
            w = 1
            while c + w < cols and grid[r, c + w] == color:
                w += 1
            # extend down
            h = 1
            while r + h < rows:
                if not np.all(grid[r + h, c:c + w] == color):
                    break
                h += 1
            rects.append((r, c, h, w, color))
            for i in range(h):
                for j in range(w):
                    visited[r + i, c + j] = True
            c += w

    for rect in rects:
        r, c, h, w, col = rect
        if col != 8:
            continue
        r2 = r + h
        if r2 + h > rows or c + w > cols:
            continue
        c_val = grid[r2, c]
        if c_val == 0 or c_val == 8:
            continue
        is_same = np.all(grid[r2:r2 + h, c:c + w] == c_val)
        if not is_same:
            continue
        # expand
        h_tot = 3 * h
        w_tot = 3 * w
        r_new = r
        c_new = c - w
        for i in range(h_tot):
            rr = r_new + i
            if rr >= rows:
                break
            for j in range(w_tot):
                cc = c_new + j
                if cc >= cols:
                    break
                if i < 2 * h:
                    if r2 <= rr < r2 + h and c <= cc < c + w:
                        continue
                    new_grid[rr, cc] = 8
                else:
                    hole_r0 = r_new + 2 * h
                    hole_c0 = c
                    if hole_r0 <= rr < hole_r0 + h and hole_c0 <= cc < hole_c0 + w:
                        new_grid[rr, cc] = 0
                    else:
                        new_grid[rr, cc] = 8

    # Step 2: Horizontal expansions
    for start_r in range(rows):
        for hh in range(1, rows - start_r + 1):
            # find common segment
            seg_start = -1
            seg_end = -1
            uniform = True
            for rr in range(start_r, start_r + hh):
                curr_start = -1
                curr_end = -1
                for cc in range(cols):
                    if grid[rr, cc] != 0:
                        if curr_start == -1:
                            curr_start = cc
                        curr_end = cc
                    elif curr_start != -1:
                        if seg_start == -1:
                            seg_start = curr_start
                            seg_end = curr_end + 1
                        else:
                            if curr_start != seg_start or curr_end + 1 != seg_end:
                                uniform = False
                                break
                        curr_start = -1
                if not uniform:
                    break
                if curr_start != -1:
                    if seg_start == -1:
                        seg_start = curr_start
                        seg_end = cols
                    else:
                        if curr_start != seg_start or cols != seg_end:
                            uniform = False
                            break
            if not uniform or seg_start == -1:
                continue
            all_start = seg_start
            all_end = seg_end
            total_w = all_end - all_start
            if total_w < hh:
                continue
            # Case 1: left active, right passive or no
            found1 = False
            w_a1 = 0
            C1 = 0
            P1 = 0
            p_w1 = 0
            for wa in range(1, total_w + 1):
                left_start = all_start
                left_end = all_start + wa
                c_test = grid[start_r, left_start]
                if c_test < 3 or c_test == 8 or c_test == 0:
                    continue
                left_uni = np.all(grid[start_r:start_r + hh, left_start:left_end] == c_test)
                if not left_uni:
                    continue
                right_start = left_end
                right_end = all_end
                pw = right_end - right_start
                if pw == 0:
                    p = 0
                else:
                    p_test = grid[start_r, right_start]
                    if p_test not in (1, 2):
                        continue
                    right_uni = np.all(grid[start_r:start_r + hh, right_start:right_end] == p_test)
                    if not right_uni:
                        continue
                    p = p_test
                found1 = True
                w_a1 = wa
                C1 = c_test
                P1 = p
                p_w1 = pw
                break  # take the largest left active
            # Case 2: left passive, right active
            found2 = False
            w_a2 = 0
            C2 = 0
            P2 = 0
            p_w2 = 0
            for pw in range(1, total_w):
                left_start = all_start
                left_end = all_start + pw
                p_test = grid[start_r, left_start]
                if p_test not in (1, 2):
                    continue
                left_uni = np.all(grid[start_r:start_r + hh, left_start:left_end] == p_test)
                if not left_uni:
                    continue
                right_start = left_end
                right_end = all_end
                wa = right_end - right_start
                if wa < 1:
                    continue
                c_test = grid[start_r, right_start]
                if c_test < 3 or c_test == 8 or c_test == 0:
                    continue
                right_uni = np.all(grid[start_r:start_r + hh, right_start:right_end] == c_test)
                if not right_uni:
                    continue
                found2 = True
                w_a2 = wa
                C2 = c_test
                P2 = p_test
                p_w2 = pw
                break  # largest left passive
            # now, if found1 or found2
            if not found1 and not found2:
                continue
            # prefer found1 (left active)
            if found1:
                w_a = w_a1
                C = C1
                P = P1
                p_w = p_w1
                passive_on_right = p_w > 0
            else:
                w_a = w_a2
                C = C2
                P = P2
                p_w = p_w2
                passive_on_right = False
            # now, common code
            s = hh
            span_w = 3 * s
            added_h_up = s
            added_h_down = s
            if p_w > 0:
                included_w = w_a + p_w if passive_on_right else w_a
                added_w_total = span_w - included_w
                if added_w_total < 0:
                    continue
                # determine extend side
                extend_left = True
                open_left = all_start >= added_w_total and np.all(grid[start_r:start_r + hh, all_start - added_w_total : all_start] == 0)
                if not open_left:
                    open_right = all_end + added_w_total <= cols and np.all(grid[start_r:start_r + hh, all_end : all_end + added_w_total] == 0)
                    if open_right:
                        extend_left = False
                    else:
                        continue
                if extend_left:
                    added_w_left = added_w_total
                    added_w_right = 0
                    bar_left = all_start - added_w_left
                else:
                    added_w_left = 0
                    added_w_right = added_w_total
                    bar_left = all_start
                # fill horizontal added
                if added_w_left > 0:
                    new_grid[start_r:start_r + hh, all_start - added_w_left : all_start] = C
                if added_w_right > 0:
                    new_grid[start_r:start_r + hh, all_end : all_end + added_w_right] = C
            else:
                # no passive, symmetric
                if w_a != hh:
                    continue  # only square
                added_w_left = hh
                added_w_right = hh
                bar_left = all_start - added_w_left
                # check open
                if all_start < added_w_left or np.any(grid[start_r:start_r + hh, bar_left : all_start] != 0):
                    continue
                if all_end + added_w_right > cols or np.any(grid[start_r:start_r + hh, all_end : all_end + added_w_right] != 0):
                    continue
                # fill
                new_grid[start_r:start_r + hh, bar_left : all_start] = C
                new_grid[start_r:start_r + hh, all_end : all_end + added_w_right] = C
            # vertical bar
            v_row_start = start_r - added_h_up
            if v_row_start < 0:
                continue
            v_height = 3 * s
            v_width = s
            if p_w > 0:
                if passive_on_right:
                    if P == 1:
                        v_col_left = all_start
                    else:
                        v_col_left = all_start + w_a
                else:
                    if P == 1:
                        v_col_left = all_start + p_w
                    else:
                        v_col_left = bar_left + span_w - v_width
            else:
                v_col_left = all_start
            if v_col_left < 0 or v_col_left + v_width > cols:
                continue
            # check and fill vertical added
            for ii in range(v_height):
                rr = v_row_start + ii
                if 0 <= rr < rows:
                    for jj in range(v_width):
                        cc = v_col_left + jj
                        if 0 <= cc < cols and grid[rr, cc] == 0:
                            new_grid[rr, cc] = C

    # Step 4: Vertical expansions (similar structure, but for columns)
    # For brevity, implement the known case upper active lower passive
    for start_c in range(cols):
        for ww in range(1, cols - start_c + 1):
            # find common row segment for these columns
            seg_start = -1
            seg_end = -1
            uniform = True
            for cc in range(start_c, start_c + ww):
                curr_start = -1
                curr_end = -1
                for rr in range(rows):
                    if grid[rr, cc] != 0:
                        if curr_start == -1:
                            curr_start = rr
                        curr_end = rr
                    elif curr_start != -1:
                        if seg_start == -1:
                            seg_start = curr_start
                            seg_end = curr_end + 1
                        else:
                            if curr_start != seg_start or curr_end + 1 != seg_end:
                                uniform = False
                                break
                        curr_start = -1
                if not uniform:
                    break
                if curr_start != -1:
                    if seg_start == -1:
                        seg_start = curr_start
                        seg_end = rows
                    else:
                        if curr_start != seg_start or rows != seg_end:
                            uniform = False
                            break
            if not uniform or seg_start == -1:
                continue
            all_start_r = seg_start
            all_end_r = seg_end
            total_h = all_end_r - all_start_r
            if total_h < ww:
                continue
            # upper active lower passive
            found = False
            h_a = 0
            C = 0
            P = 0
            p_h = 0
            for ha in range(1, total_h + 1):
                upper_start = all_start_r
                upper_end = all_start_r + ha
                c_test = grid[upper_start, start_c]
                if c_test < 3 or c_test == 8 or c_test == 0:
                    continue
                upper_uni = True
                for rr in range(upper_start, upper_end):
                    for cc in range(start_c, start_c + ww):
                        if grid[rr, cc] != c_test:
                            upper_uni = False
                            break
                    if not upper_uni:
                        break
                if not upper_uni:
                    continue
                lower_start = upper_end
                lower_end = all_end_r
                ph = lower_end - lower_start
                if ph == 0:
                    continue  # no passive for now
                p_test = grid[lower_start, start_c]
                if p_test not in (1, 2):
                    continue
                lower_uni = True
                for rr in range(lower_start, lower_end):
                    for cc in range(start_c, start_c + ww):
                        if grid[rr, cc] != p_test:
                            lower_uni = False
                            break
                    if not lower_uni:
                        break
                if not lower_uni:
                    continue
                found = True
                h_a = ha
                C = c_test
                P = p_test
                p_h = ph
                break
            if not found:
                continue
            # check adjacent_8 (analog)
            adjacent_8 = False
            # left
            if start_c > 0:
                col_left = start_c - 1
                if np.any(grid[all_start_r:all_end_r, col_left] == 8):
                    adjacent_8 = True
            # right
            if start_c + ww < cols:
                col_right = start_c + ww
                if np.any(grid[all_start_r:all_end_r, col_right] == 8):
                    adjacent_8 = True
            # up
            if all_start_r > 0:
                row_up = all_start_r - 1
                if np.any(grid[row_up, start_c:start_c + ww] == 8):
                    adjacent_8 = True
            # down
            if all_end_r < rows:
                row_down = all_end_r
                if np.any(grid[row_down, start_c:start_c + ww] == 8):
                    adjacent_8 = True
            if adjacent_8:
                continue
            # expand vertical
            s = ww
            span_v = 3 * s
            added_h_down = s
            added_h_up = 0
            # horizontal bar
            span_w = 3 * s
            height_bar = s
            included_h = h_a + p_h
            added_w_left = included_h  # total_h
            added_w_right = 0
            # place at bottom since P=2
            bar_r_start = all_start_r + h_a  # lower start
            bar_r_end = bar_r_start + height_bar
            bar_c_left = start_c - added_w_left
            if bar_c_left < 0:
                continue
            # check open left
            open_left = True
            for i in range(added_w_left):
                cc = start_c - 1 - i
                if np.any(grid[bar_r_start:bar_r_end, cc] != 0):
                    open_left = False
                    break
            if not open_left:
                continue
            # fill added left
            new_grid[bar_r_start:bar_r_end, bar_c_left:start_c] = C
            # vertical added down
            down_r_start = all_end_r
            down_r_end = down_r_start + added_h_down
            if down_r_end > rows:
                continue
            open_down = np.all(grid[down_r_start:down_r_end, start_c:start_c + ww] == 0)
            if open_down:
                new_grid[down_r_start:down_r_end, start_c:start_c + ww] = C

    return new_grid.tolist()