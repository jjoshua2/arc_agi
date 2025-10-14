import numpy as np
from collections import defaultdict

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = np.array(grid_lst)
    nrows, ncols = grid.shape

    # Find full rows: rows with no 0
    full_rows = [i for i in range(nrows) if np.all(grid[i] != 0)]

    if not full_rows:
        return [[0] * 3 for _ in range(3)]

    large_nk = len(full_rows)

    # Find background color: the color that appears in full rows without insertions
    # Find a full row with minimal unique non-0 colors, assume 1
    bg = None
    for i in full_rows:
        s = set(grid[i][grid[i] != 0])
        if len(s) == 1:
            bg = list(s)[0]
            break
    if bg is None:
        # Otherwise, the most common non-0 in full rows
        count = defaultdict(int)
        for i in full_rows:
            for v in grid[i]:
                if v != 0:
                    count[v] += 1
        if count:
            bg = max(count, key=count.get)

    if bg is None:
        bg = 0  # fallback

    # Find line columns: columns that have bg in some full row, and in non-full rows have 0 or bg
    line_cols = []
    for j in range(ncols):
        is_line = all(grid[i, j] == 0 or grid[i, j] == bg for i in range(nrows) if i not in set(full_rows))
        if is_line and any(grid[i, j] == bg for i in full_rows):
            line_cols.append(j)

    large_nm = len(line_cols)
    if large_nm == 0:
        line_cols = list(range(ncols))
        large_nm = ncols

    # Build large_grid: large_nk x large_nm, value if != bg else 0
    large_grid = np.zeros((large_nk, large_nm), dtype=int)
    for kk in range(large_nk):
        i = full_rows[kk]
        for mm in range(large_nm):
            j = line_cols[mm]
            v = grid[i, j]
            large_grid[kk, mm] = v if v != bg else 0

    # Now, group sizes
    def get_group_bounds(n):
        gs = n // 3
        rem = n % 3
        bounds = [0]
        for i in range(3):
            size = gs + (1 if i < rem else 0)
            bounds.append(bounds[-1] + size)
        return bounds

    k_bounds = get_group_bounds(large_nk)
    m_bounds = get_group_bounds(large_nm)

    output = [[0 for _ in range(3)] for _ in range(3)]
    previous_row = [0, 0, 0]

    for r in range(3):
        start_k = k_bounds[r]
        end_k = k_bounds[r + 1] - 1
        size_k = end_k - start_k + 1

        start_ms = m_bounds
        temp_row = [0] * 3
        has_resolved_mixed = False

        for c in range(3):
            start_m = m_bounds[c]
            end_m = m_bounds[c + 1] - 1
            size_m = end_m - start_m + 1

            color_count = defaultdict(int)
            has_same_k_multiple = False

            for kk in range(start_k, end_k + 1):
                appears = set()
                for mm in range(start_m, end_m + 1):
                    v = large_grid[kk, mm]
                    if v != 0:
                        appears.add(v)
                if len(appears) > 1:
                    has_same_k_multiple = True
                for v in appears:
                    color_count[v] += 1

            if has_same_k_multiple:
                temp_row[c] = 0
            else:
                if len(color_count) == 0:
                    temp_row[c] = 0
                elif len(color_count) == 1:
                    temp_row[c] = next(iter(color_count))
                else:
                    max_s = max(color_count.values())
                    candidates = [v for v, cnt in color_count.items() if cnt == max_s]
                    if len(candidates) == 1:
                        temp_row[c] = candidates[0]
                        has_resolved_mixed = True
                    else:
                        temp_row[c] = 0

        # Post processing
        # Reverse if resolved mixed
        if has_resolved_mixed:
            temp_row = temp_row[::-1]

        # Move for adjacent different in col0 and col1
        if len(temp_row) >= 2 and temp_row[0] != 0 and temp_row[1] != 0 and temp_row[0] != temp_row[1]:
            temp_row[2] = temp_row[1]
            temp_row[1] = 0

        # Shift left if col1 and col2 same non-zero
        if len(temp_row) >= 3 and temp_row[1] != 0 and temp_row[2] != 0 and temp_row[1] == temp_row[2]:
            temp_row[0] = temp_row[1]
            temp_row[2] = 0

        # Fill for r==2 if single color and span_m >= threshold
        if r == 2:
            non_zero_cs = [cc for cc in range(3) if temp_row[cc] != 0]
            if non_zero_cs:
                colors = set(temp_row[cc] for cc in non_zero_cs)
                if len(colors) == 1:
                    color = list(colors)[0]
                    min_m = large_nm
                    max_m = -1
                    for mm in range(large_nm):
                        appears_in_k = False
                        for kk in range(start_k, end_k + 1):
                            if large_grid[kk, mm] == color:
                                appears_in_k = True
                                break
                        if appears_in_k:
                            min_m = min(min_m, mm)
                            max_m = max(max_m, mm)
                    if min_m <= max_m:
                        span_m = max_m - min_m + 1
                        threshold = large_nm // 2
                        if span_m >= threshold:
                            for cc in range(3):
                                if temp_row[cc] == 0:
                                    temp_row[cc] = color

        # Single shift for r==2 if single non-zero in col1
        if r == 2:
            non_zero_count = sum(1 for v in temp_row if v != 0)
            if non_zero_count == 1 and temp_row[1] != 0:
                temp_row[2] = temp_row[1]
                temp_row[1] = 0

        # Copy if all zero and r>0
        if all(v == 0 for v in temp_row) and r > 0:
            temp_row = previous_row[:]

        output[r] = temp_row
        previous_row = temp_row[:]

    return output