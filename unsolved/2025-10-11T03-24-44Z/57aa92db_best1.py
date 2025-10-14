def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    rows = len(grid_lst)
    if rows == 0:
        return []
    cols = len(grid_lst[0])
    output = [row[:] for row in grid_lst]

    # Find connected components (4-connected, same color > 0)
    visited = [[False] * cols for _ in range(rows)]
    components = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(rows):
        for c in range(cols):
            if output[r][c] > 0 and not visited[r][c]:
                color = output[r][c]
                stack = [(r, c)]
                visited[r][c] = True
                min_r = max_r = r
                min_c = max_c = c
                while stack:
                    cr, cc = stack.pop()
                    min_r = min(min_r, cr)
                    max_r = max(max_r, cr)
                    min_c = min(min_c, cc)
                    max_c = max(max_c, cc)
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and output[nr][nc] == color:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                components.append({
                    'color': color,
                    'minr': min_r,
                    'maxr': max_r,
                    'minc': min_c,
                    'maxc': max_c
                })

    # Process horizontal pairs
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            comp1 = components[i]
            comp2 = components[j]
            if comp1['minr'] != comp2['minr'] or comp1['maxr'] != comp2['maxr']:
                continue
            h = comp1['maxr'] - comp1['minr'] + 1
            w1 = comp1['maxc'] - comp1['minc'] + 1
            w2 = comp2['maxc'] - comp2['minc'] + 1
            if w1 != h or w2 != h:
                continue

            # Determine left and right comp
            if comp1['maxc'] + 1 == comp2['minc']:
                left_comp = comp1
                right_comp = comp2
            elif comp2['maxc'] + 1 == comp1['minc']:
                left_comp = comp2
                right_comp = comp1
            else:
                continue

            c_left = left_comp['color']
            c_right = right_comp['color']
            if c_left > c_right:
                higher = left_comp
                lower = right_comp
                higher_is_left = True
                amount = h
            elif c_right > c_left:
                higher = right_comp
                lower = left_comp
                higher_is_left = False
                amount = 2 * h
            else:
                continue

            minr = higher['minr']
            maxr = higher['maxr']
            if higher_is_left:
                extend_start = higher['minc'] - amount
                extend_end = higher['minc']
            else:
                extend_start = higher['maxc'] + 1
                extend_end = higher['maxc'] + 1 + amount

            # Fill horizontal extension
            for r in range(minr, maxr + 1):
                for c in range(extend_start, extend_end):
                    if 0 <= c < cols and output[r][c] == 0:
                        output[r][c] = higher['color']

            # Vertical extension
            up_start = minr - h
            up_end = minr
            down_start = maxr + 1
            down_end = maxr + 1 + h
            if higher_is_left:
                if h == 1:
                    v_min = lower['minc']
                    v_max = lower['maxc'] + 1
                else:
                    v_min = higher['minc']
                    v_max = higher['maxc'] + 1
            else:
                higher_width = higher['maxc'] - higher['minc'] + 1
                far_start = higher['maxc'] + amount - higher_width + 1
                far_end = higher['maxc'] + amount + 1
                v_min = max(0, far_start)
                v_max = min(cols, far_end)
                if v_min >= v_max:
                    continue  # no vertical

            for rr in list(range(up_start, up_end)) + list(range(down_start, down_end)):
                if 0 <= rr < rows:
                    for cc in range(v_min, v_max):
                        if 0 <= cc < cols and output[rr][cc] == 0:
                            output[rr][cc] = higher['color']

    # Process vertical pairs (only higher upper for now)
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            comp1 = components[i]
            comp2 = components[j]
            if comp1['minc'] != comp2['minc'] or comp1['maxc'] != comp2['maxc']:
                continue
            w = comp1['maxc'] - comp1['minc'] + 1
            if comp1['maxr'] + 1 == comp2['minr']:
                upper_comp = comp1
                lower_comp = comp2
            elif comp2['maxr'] + 1 == comp1['minr']:
                upper_comp = comp2
                lower_comp = comp1
            else:
                continue
            h_u = upper_comp['maxr'] - upper_comp['minr'] + 1
            h_l = lower_comp['maxr'] - lower_comp['minr'] + 1
            if h_u != h_l or w != h_u:
                continue
            n = h_u
            c_u = upper_comp['color']
            c_l = lower_comp['color']
            if c_u <= c_l:
                continue  # only higher upper
            min_col = upper_comp['minc']
            max_col = upper_comp['maxc']

            # blocked
            blocked_left = False
            blocked_right = False
            for k in range(len(components)):
                if k == i or k == j:
                    continue
                comp_k = components[k]
                c_k = comp_k['color']
                if c_k > c_u:
                    if comp_k['minc'] < min_col:
                        blocked_left = True
                    if comp_k['maxc'] > max_col:
                        blocked_right = True

            t_left = n if not blocked_left else 0
            t_right = n if not blocked_right else 0
            total_t = t_left + t_right
            needed = 2 * n
            if total_t < needed:
                diff = needed - total_t
                if t_left > 0:
                    t_left += diff
                elif t_right > 0:
                    t_right += diff

            left_start = min_col - t_left
            left_end = min_col
            right_start = max_col + 1
            right_end = max_col + t_right + 1
            side_cols_left = range(left_start, left_end) if t_left > 0 else []
            side_cols_right = range(right_start, right_end) if t_right > 0 else []
            all_side = list(side_cols_left) + list(side_cols_right)
            is_symmetric = t_left > 0 and t_right > 0

            minr_u = upper_comp['minr']
            maxr_u = upper_comp['maxr']
            minr_l = lower_comp['minr']
            maxr_l = lower_comp['maxr']
            upper_r = range(minr_u, maxr_u + 1)
            middle_r = range(minr_l, maxr_l + 1)
            lower_r_start = maxr_l + 1
            lower_r_end = maxr_l + 1 + n
            lower_r = range(lower_r_start, lower_r_end)

            # upper fill if symmetric
            if is_symmetric:
                for r in upper_r:
                    if 0 <= r < rows:
                        for c in all_side:
                            if 0 <= c < cols and output[r][c] == 0:
                                output[r][c] = c_u

            # middle always
            for r in middle_r:
                if 0 <= r < rows:
                    for c in all_side:
                        if 0 <= c < cols and output[r][c] == 0:
                            output[r][c] = c_u

            # lower
            if is_symmetric:
                for r in lower_r:
                    if 0 <= r < rows:
                        for c in all_side:
                            if 0 <= c < cols and output[r][c] == 0:
                                output[r][c] = c_u
            else:
                center_c = range(min_col, max_col + 1)
                for r in lower_r:
                    if 0 <= r < rows:
                        for c in center_c:
                            if 0 <= c < cols and output[r][c] == 0:
                                output[r][c] = c_u

    return output