from collections import Counter
import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    rows = len(grid)
    cols = len(grid[0])
    flat = [cell for row in grid for cell in row]
    counter = Counter(flat)
    possible_c = [c for c in counter if c != 0]
    if not possible_c:
        return [row[:] for row in grid]
    C = max(possible_c, key=lambda x: counter[x])

    has_C = [False] * cols
    for c in range(cols):
        for r in range(rows):
            if grid[r][c] == C:
                has_C[c] = True
                break

    free = [c for c in range(cols) if not has_C[c]]

    if not free:
        return [row[:] for row in grid]

    # Find longest consecutive free columns
    free.sort()
    max_len = 1
    max_start = 0
    current_start = 0
    starts = []
    lens = []
    for i in range(1, len(free)):
        if free[i] == free[i-1] + 1:
            continue
        else:
            cur_len = i - current_start
            starts.append(current_start)
            lens.append(cur_len)
            if cur_len > max_len:
                max_len = cur_len
                max_start = current_start
            current_start = i
    cur_len = len(free) - current_start
    starts.append(current_start)
    lens.append(cur_len)
    if cur_len > max_len:
        max_len = cur_len
        max_start = current_start

    # If multiple with max_len, take leftmost
    candidates = [starts[j] for j in range(len(lens)) if lens[j] == max_len]
    max_start = min(candidates)

    L = free[max_start]
    R = free[max_start + max_len - 1]

    output = [row[:] for row in grid]

    # Main fill
    main_cols = range(L + 1, R)
    for r in range(rows):
        for cc in main_cols:
            if output[r][cc] == 0:
                output[r][cc] = 3

    # Left extension
    if L >= 0:
        left_range = range(0, L + 1)
        no_C_left = [True] * rows
        for r in range(rows):
            for cc in left_range:
                if grid[r][cc] == C:
                    no_C_left[r] = False
                    break

        i = 0
        while i < rows:
            if no_C_left[i]:
                s = i
                while i < rows and no_C_left[i]:
                    i += 1
                t = i - 1
                len_b = t - s + 1
                if len_b >= 3:
                    fill_s = s + 1
                    fill_t = t - 1
                    for rr in range(fill_s, fill_t + 1):
                        for cc in left_range:
                            if output[rr][cc] == 0:
                                output[rr][cc] = 3
            else:
                i += 1

    # Right extension
    if R < cols - 1:
        right_range = range(R, cols)
        no_C_right = [True] * rows
        for r in range(rows):
            for cc in right_range:
                if grid[r][cc] == C:
                    no_C_right[r] = False
                    break

        i = 0
        while i < rows:
            if no_C_right[i]:
                s = i
                while i < rows and no_C_right[i]:
                    i += 1
                t = i - 1
                len_b = t - s + 1
                if len_b >= 3:
                    fill_s = s + 1
                    fill_t = t - 1
                    for rr in range(fill_s, fill_t + 1):
                        for cc in right_range:
                            if output[rr][cc] == 0:
                                output[rr][cc] = 3
            else:
                i += 1

    return output