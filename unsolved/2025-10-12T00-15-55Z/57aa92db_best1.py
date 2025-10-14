from collections import deque
import copy

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return copy.deepcopy(grid)
    rows = len(grid)
    cols = len(grid[0])
    output = copy.deepcopy(grid)
    
    # Get components
    visited = [[False] * cols for _ in range(rows)]
    components = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                C = grid[r][c]
                comp_cells = []
                q = deque([(r, c)])
                visited[r][c] = True
                min_r, max_r = r, r
                min_c, max_c = c, c
                comp_cells.append((r, c))
                
                while q:
                    x, y = q.popleft()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and grid[nx][ny] == C:
                            visited[nx][ny] = True
                            q.append((nx, ny))
                            comp_cells.append((nx, ny))
                            min_r = min(min_r, nx)
                            max_r = max(max_r, nx)
                            min_c = min(min_c, ny)
                            max_c = max(max_c, ny)
                
                # Check if rectangle
                h_comp = max_r - min_r + 1
                w_comp = max_c - min_c + 1
                if len(comp_cells) == h_comp * w_comp:
                    is_rect = True
                    for i in range(min_r, max_r + 1):
                        for j in range(min_c, max_c + 1):
                            if grid[i][j] != C:
                                is_rect = False
                                break
                        if not is_rect:
                            break
                    if is_rect:
                        components.append((C, min_r, max_r, min_c, max_c))
    
    # Process each potential main
    for idx_main in range(len(components)):
        C, min_r, max_r, min_c, max_c = components[idx_main]
        if C in {1, 2}:
            continue
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        
        # Check for non-special same-size below for framing
        framed = False
        for idx_tail in range(len(components)):
            if idx_main == idx_tail:
                continue
            t_C, t_min_r, t_max_r, t_min_c, t_max_c = components[idx_tail]
            t_h = t_max_r - t_min_r + 1
            if t_min_r == max_r + 1 and t_min_c == min_c and t_max_c == max_c and t_h == h and t_C not in {0, 1, 2}:
                # Frame below
                framed = True
                left_min = min_c - h
                right_max = max_c + h
                # Upper full
                for rr in range(min_r, max_r + 1):
                    for cc in range(left_min, right_max + 1):
                        if 0 <= cc < cols and output[rr][cc] == 0:
                            output[rr][cc] = C
                # Middle sides
                for rr in range(t_min_r, t_max_r + 1):
                    for cc in range(left_min, min_c):
                        if 0 <= cc < cols and output[rr][cc] == 0:
                            output[rr][cc] = C
                    for cc in range(max_c + 1, right_max + 1):
                        if 0 <= cc < cols and output[rr][cc] == 0:
                            output[rr][cc] = C
                # Lower sides
                lower_end = t_max_r + h
                for rr in range(t_max_r + 1, lower_end + 1):
                    if 0 <= rr < rows:
                        for cc in range(left_min, min_c):
                            if 0 <= cc < cols and output[rr][cc] == 0:
                                output[rr][cc] = C
                        for cc in range(max_c + 1, right_max + 1):
                            if 0 <= cc < cols and output[rr][cc] == 0:
                                output[rr][cc] = C
                break
        
        if framed:
            continue
        
        # Check for special tail
        tail_info = None
        dir_attach = None
        for idx_tail in range(len(components)):
            if idx_main == idx_tail:
                continue
            t_C, t_min_r, t_max_r, t_min_c, t_max_c = components[idx_tail]
            if t_C not in {1, 2}:
                continue
            # horizontal right
            if t_min_c == max_c + 1 and max(t_min_r, min_r) <= min(t_max_r, max_r):
                tail_info = (t_C, t_min_r, t_max_r, t_min_c, t_max_c)
                dir_attach = 'right'
                break
            # left
            if t_max_c + 1 == min_c and max(t_min_r, min_r) <= min(t_max_r, max_r):
                tail_info = (t_C, t_min_r, t_max_r, t_min_c, t_max_c)
                dir_attach = 'left'
                break
            # below
            if t_min_r == max_r + 1 and max(t_min_c, min_c) <= min(t_max_c, max_c):
                tail_info = (t_C, t_min_r, t_max_r, t_min_c, t_max_c)
                dir_attach = 'below'
                break
            # above
            if t_max_r + 1 == min_r and max(t_min_c, min_c) <= min(t_max_c, max_c):
                tail_info = (t_C, t_min_r, t_max_r, t_min_c, t_max_c)
                dir_attach = 'above'
                break
        
        if tail_info is None:
            continue  # no tail, no growth
        
        t_C, t_min_r, t_max_r, t_min_c, t_max_c = tail_info
        t_h = t_max_r - t_min_r + 1
        t_w = t_max_c - t_min_c + 1
        
        if dir_attach in ['left', 'right']:
            # horizontal special tail
            if h == 1:
                main_r = min_r
                if dir_attach == 'left':  # away right, extend 2
                    for k in range(1, 3):
                        cc = max_c + k
                        if 0 <= cc < cols and output[main_r][cc] == 0:
                            output[main_r][cc] = C
                    far_cc = max_c + 2
                    if 0 <= far_cc < cols:
                        # up
                        up_rr = main_r - 1
                        if 0 <= up_rr < rows and output[up_rr][far_cc] == 0:
                            output[up_rr][far_cc] = C
                        # down
                        down_rr = main_r + 1
                        if 0 <= down_rr < rows and output[down_rr][far_cc] == 0:
                            output[down_rr][far_cc] = C
                else:  # right, away left, extend 1
                    left_cc = min_c - 1
                    if 0 <= left_cc < cols and output[main_r][left_cc] == 0:
                        output[main_r][left_cc] = C
                    tail_left_c = t_min_c  # = max_c +1
                    # up
                    up_rr = main_r - 1
                    if 0 <= up_rr < rows and output[up_rr][tail_left_c] == 0:
                        output[up_rr][tail_left_c] = C
                    # down
                    down_rr = main_r + 1
                    if 0 <= down_rr < rows and output[down_rr][tail_left_c] == 0:
                        output[down_rr][tail_left_c] = C
            else:
                # h >1, cross
                # horizontal away
                if dir_attach == 'left':  # away right
                    h_start_c = max_c + 1
                    h_end_c = max_c + h
                    for cc in range(h_start_c, h_end_c + 1):
                        if 0 <= cc < cols:
                            for rr in range(min_r, max_r + 1):
                                if output[rr][cc] == 0:
                                    output[rr][cc] = C
                else:  # away left
                    h_start_c = min_c - h
                    h_end_c = min_c - 1
                    for cc in range(h_start_c, h_end_c + 1):
                        if 0 <= cc < cols:
                            for rr in range(min_r, max_r + 1):
                                if output[rr][cc] == 0:
                                    output[rr][cc] = C
                # up
                u_start_r = min_r - h
                u_end_r = min_r - 1
                for rr in range(u_start_r, u_end_r + 1):
                    if 0 <= rr < rows:
                        for cc in range(min_c, max_c + 1):
                            if output[rr][cc] == 0:
                                output[rr][cc] = C
                # down
                d_start_r = max_r + 1
                d_end_r = max_r + h
                for rr in range(d_start_r, d_end_r + 1):
                    if 0 <= rr < rows:
                        for cc in range(min_c, max_c + 1):
                            if output[rr][cc] == 0:
                                output[rr][cc] = C
        elif dir_attach == 'below':
            # special vertical below
            # horizontal left in tail rows by 2*h
            left_start = t_min_c - 2 * h
            for cc in range(left_start, t_min_c):
                if 0 <= cc < cols:
                    for rr in range(t_min_r, t_max_r + 1):
                        if output[rr][cc] == 0:
                            output[rr][cc] = C
            # down beyond
            d_start_r = t_max_r + 1
            d_end_r = t_max_r + h
            for rr in range(d_start_r, d_end_r + 1):
                if 0 <= rr < rows:
                    for cc in range(min_c, max_c + 1):
                        if output[rr][cc] == 0:
                            output[rr][cc] = C
        elif dir_attach == 'above':
            # special vertical above, symmetric
            # horizontal right in tail rows by 2*h
            right_start = t_max_c + 1
            right_end = t_max_c + 2 * h
            for cc in range(right_start, right_end + 1):
                if 0 <= cc < cols:
                    for rr in range(t_min_r, t_max_r + 1):
                        if output[rr][cc] == 0:
                            output[rr][cc] = C
            # up beyond
            u_start_r = t_min_r - h
            u_end_r = t_min_r - 1
            for rr in range(u_start_r, u_end_r + 1):
                if 0 <= rr < rows:
                    for cc in range(min_c, max_c + 1):
                        if output[rr][cc] == 0:
                            output[rr][cc] = C
    
    return output