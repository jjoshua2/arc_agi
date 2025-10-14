def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    visited = [[False] * cols for _ in range(rows)]
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    rects = []
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 0 and not visited[i][j]:
                color = grid[i][j]
                component = []
                stack = [(i, j)]
                visited[i][j] = True
                min_r = max_r = i
                min_c = max_c = j
                while stack:
                    x, y = stack.pop()
                    component.append((x, y))
                    min_r = min(min_r, x)
                    max_r = max(max_r, x)
                    min_c = min(min_c, y)
                    max_c = max(max_c, y)
                    for dx, dy in dirs:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and grid[nx][ny] == color:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
                h_comp = max_r - min_r + 1
                w_comp = max_c - min_c + 1
                if len(component) == h_comp * w_comp:
                    rects.append((min_r, max_r, min_c, max_c, color, h_comp, w_comp))
    rects.sort(key=lambda x: (x[0], x[2]))
    for minr, maxr, minc, maxc, color, h, w in rects:
        if color == 1 or color == 2:
            continue
        if h == 1 and w == 1:
            r = minr
            c = minc
            blocked_left = c > 0 and grid[r][c - 1] == 2
            blocked_right = c + 1 < cols and grid[r][c + 1] == 2
            if blocked_left:
                for dc in [1, 2]:
                    nc = c + dc
                    if nc < cols and output[r][nc] == 0:
                        output[r][nc] = color
                for dr in [-1, 1]:
                    nr = r + dr
                    if 0 <= nr < rows and c + 2 < cols and output[nr][c + 2] == 0:
                        output[nr][c + 2] = color
                continue
            elif blocked_right:
                nc = c - 1
                if nc >= 0 and output[r][nc] == 0:
                    output[r][nc] = color
                bc = c + 1
                for dr in [-1, 1]:
                    nr = r + dr
                    if 0 <= nr < rows and output[nr][bc] == 0:
                        output[nr][bc] = color
                continue
            else:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and output[nr][nc] == 0:
                        output[nr][nc] = color
                continue
        # larger
        special_done = False
        # special below 2
        if color != 8:
            below_hh = h
            b_minr = maxr + 1
            b_maxr = maxr + below_hh
            if b_maxr < rows:
                all_two = True
                for rr in range(b_minr, b_maxr + 1):
                    for cc in range(minc, maxc + 1):
                        if grid[rr][cc] != 2:
                            all_two = False
                            break
                    if not all_two:
                        break
                if all_two and (maxc - minc + 1) == w:
                    # special below 2
                    # left in below rows
                    left_l = 2 * w
                    left_minc_ = minc - left_l
                    all_zero_left = True
                    if left_minc_ >= 0:
                        for rr in range(b_minr, b_maxr + 1):
                            for cc in range(left_minc_, minc):
                                if grid[rr][cc] != 0:
                                    all_zero_left = False
                                    break
                            if not all_zero_left:
                                break
                        if all_zero_left:
                            for rr in range(b_minr, b_maxr + 1):
                                for cc in range(max(0, left_minc_), minc):
                                    output[rr][cc] = color
                    # down beyond
                    down_l = h
                    d_minr = b_maxr + 1
                    d_maxr = b_maxr + down_l
                    all_zero_down = True
                    if d_maxr < rows:
                        for rr in range(d_minr, d_maxr + 1):
                            for cc in range(minc, maxc + 1):
                                if grid[rr][cc] != 0:
                                    all_zero_down = False
                                    break
                        if all_zero_down:
                            for rr in range(d_minr, d_maxr + 1):
                                for cc in range(minc, maxc + 1):
                                    output[rr][cc] = color
                    special_done = True
        if not special_done and color == 8 and h == w:
            side = h
            b_minr = maxr + 1
            b_maxr = maxr + side
            if b_maxr < rows:
                all_four = True
                for rr in range(b_minr, b_maxr + 1):
                    for cc in range(minc, maxc + 1):
                        if grid[rr][cc] != 4:
                            all_four = False
                            break
                if all_four:
                    # special frame
                    l_side = side
                    left_minc_ = minc - l_side
                    right_maxc_ = maxc + l_side
                    frame_maxr_ = maxr + 2 * side
                    f_minr = minr
                    f_maxr = min(frame_maxr_, rows - 1)
                    # left
                    all_zero_left = True
                    l_min_c = max(0, left_minc_)
                    for cc in range(l_min_c, minc):
                        for rr in range(f_minr, f_maxr + 1):
                            if grid[rr][cc] != 0:
                                all_zero_left = False
                                break
                        if not all_zero_left:
                            break
                    if all_zero_left:
                        for cc in range(l_min_c, minc):
                            for rr in range(f_minr, f_maxr + 1):
                                output[rr][cc] = color
                    # right
                    all_zero_right = True
                    r_min_c = maxc + 1
                    r_max_c = min(cols - 1, right_maxc_)
                    for cc in range(r_min_c, r_max_c + 1):
                        for rr in range(f_minr, f_maxr + 1):
                            if grid[rr][cc] != 0:
                                all_zero_right = False
                                break
                    if all_zero_right:
                        for cc in range(r_min_c, r_max_c + 1):
                            for rr in range(f_minr, f_maxr + 1):
                                output[rr][cc] = color
                    special_done = True
        if not special_done and color == 4 and h == w:
            side = h
            a_maxr = minr - 1
            a_minr = minr - side
            if a_minr >= 0:
                all_eight = True
                for rr in range(a_minr, a_maxr + 1):
                    for cc in range(minc, maxc + 1):
                        if grid[rr][cc] != 8:
                            all_eight = False
                            break
                if all_eight:
                    continue  # skip normal
        if not special_done:
            # normal arms
            # up l = h
            l = h
            up_minr_ = minr - l
            if up_minr_ >= 0:
                all_zero = True
                for rr in range(up_minr_, minr):
                    for cc in range(minc, maxc + 1):
                        if grid[rr][cc] != 0:
                            all_zero = False
                            break
                if all_zero:
                    for rr in range(up_minr_, minr):
                        for cc in range(minc, maxc + 1):
                            output[rr][cc] = color
            # down l = h
            l = h
            down_maxr_ = maxr + l
            if down_maxr_ < rows:
                all_zero = True
                for rr in range(maxr + 1, down_maxr_ + 1):
                    for cc in range(minc, maxc + 1):
                        if grid[rr][cc] != 0:
                            all_zero = False
                            break
                if all_zero:
                    for rr in range(maxr + 1, down_maxr_ + 1):
                        for cc in range(minc, maxc + 1):
                            output[rr][cc] = color
            # left l = w
            l = w
            left_minc_ = minc - l
            if left_minc_ >= 0:
                all_zero = True
                for cc in range(left_minc_, minc):
                    for rr in range(minr, maxr + 1):
                        if grid[rr][cc] != 0:
                            all_zero = False
                            break
                if all_zero:
                    for cc in range(left_minc_, minc):
                        for rr in range(minr, maxr + 1):
                            output[rr][cc] = color
            # right l = w
            l = w
            right_maxc_ = maxc + l
            if right_maxc_ < cols:
                all_zero = True
                for cc in range(maxc + 1, right_maxc_ + 1):
                    for rr in range(minr, maxr + 1):
                        if grid[rr][cc] != 0:
                            all_zero = False
                            break
                if all_zero:
                    for cc in range(maxc + 1, right_maxc_ + 1):
                        for rr in range(minr, maxr + 1):
                            output[rr][cc] = color
    return output