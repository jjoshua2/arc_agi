import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    h, w = grid.shape

    hor_lines = []
    for r in range(h):
        c = 0
        while c < w:
            if grid[r, c] not in (2, 5):
                c += 1
                continue
            start = c
            red_cnt = 0
            while c < w and grid[r, c] in (2, 5):
                if grid[r, c] == 2:
                    red_cnt += 1
                c += 1
            end = c - 1
            if red_cnt >= 2:
                hor_lines.append((r, start, end))

    vert_lines = []
    for c in range(w):
        r = 0
        while r < h:
            if grid[r, c] not in (2, 5):
                r += 1
                continue
            start = r
            red_cnt = 0
            while r < h and grid[r, c] in (2, 5):
                if grid[r, c] == 2:
                    red_cnt += 1
                r += 1
            end = r - 1
            if red_cnt >= 2:
                vert_lines.append((c, start, end))

    for h_r, h_start, h_end in hor_lines:
        for v_c, v_start, v_end in vert_lines:
            if v_start <= h_r <= v_end and h_start <= v_c <= h_end:
                center_r = h_r
                center_c = v_c

                far_right = 0
                for m in range(1, w - center_c):
                    pc = center_c + m
                    if grid[center_r, pc] == 0:
                        break
                    if grid[center_r, pc] == 2:
                        far_right = m

                far_left = 0
                for m in range(1, center_c + 1):
                    pc = center_c - m
                    if grid[center_r, pc] == 0:
                        break
                    if grid[center_r, pc] == 2:
                        far_left = m

                far_up = 0
                for m in range(1, center_r + 1):
                    pr = center_r - m
                    if grid[pr, center_c] == 0:
                        break
                    if grid[pr, center_c] == 2:
                        far_up = m

                far_down = 0
                for m in range(1, h - center_r):
                    pr = center_r + m
                    if grid[pr, center_c] == 0:
                        break
                    if grid[pr, center_c] == 2:
                        far_down = m

                L = max(far_right, far_left, far_up, far_down)
                if L == 0:
                    continue

                if grid[center_r, center_c] == 5:
                    grid[center_r, center_c] = 8

                # fill right
                for m in range(1, L + 1):
                    pc = center_c + m
                    if pc >= w:
                        break
                    if grid[center_r, pc] == 0:
                        break
                    if grid[center_r, pc] == 5:
                        grid[center_r, pc] = 8

                # fill left
                for m in range(1, L + 1):
                    pc = center_c - m
                    if pc < 0:
                        break
                    if grid[center_r, pc] == 0:
                        break
                    if grid[center_r, pc] == 5:
                        grid[center_r, pc] = 8

                # fill up
                for m in range(1, L + 1):
                    pr = center_r - m
                    if pr < 0:
                        break
                    if grid[pr, center_c] == 0:
                        break
                    if grid[pr, center_c] == 5:
                        grid[pr, center_c] = 8

                # fill down
                for m in range(1, L + 1):
                    pr = center_r + m
                    if pr >= h:
                        break
                    if grid[pr, center_c] == 0:
                        break
                    if grid[pr, center_c] == 5:
                        grid[pr, center_c] = 8

    return grid.tolist()