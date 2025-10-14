def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid:
        return []
    rows = len(grid)
    if rows == 0:
        return []
    cols = len(grid[0])
    # Find terminals
    terminals = [(i, j) for i in range(rows) for j in range(cols) if grid[i][j] == 2]
    if not terminals:
        return [row[:] for row in grid]
    # Check for horizontal spine
    row_counts = {}
    for r, c in terminals:
        row_counts[r] = row_counts.get(r, 0) + 1
    horizontal_spine = None
    for r in sorted(row_counts):
        if row_counts[r] >= 2:
            horizontal_spine = r
            break
    if horizontal_spine is not None:
        # Horizontal spine
        spine_r = horizontal_spine
        spine_terms_cs = [c for r, c in terminals if r == spine_r]
        min_c = min(spine_terms_cs)
        max_c = max(spine_terms_cs)
        output = [row[:] for row in grid]
        # Fill spine horizontal
        for c in range(min_c, max_c + 1):
            if output[spine_r][c] != 2:
                output[spine_r][c] = 3
        # Connect other terminals vertically
        for tr, tc in terminals:
            if tr == spine_r:
                continue
            start_r = min(tr, spine_r)
            end_r = max(tr, spine_r)
            for rr in range(start_r, end_r + 1):
                if output[rr][tc] != 2:
                    output[rr][tc] = 3
        return output
    else:
        # Check for vertical spine
        col_counts = {}
        for r, c in terminals:
            col_counts[c] = col_counts.get(c, 0) + 1
        vertical_spine = None
        for c in sorted(col_counts):
            if col_counts[c] >= 2:
                vertical_spine = c
                break
        if vertical_spine is None:
            return [row[:] for row in grid]
        # Vertical spine
        spine_c = vertical_spine
        spine_terms_rs = [r for r, c in terminals if c == spine_c]
        min_r = min(spine_terms_rs)
        max_r = max(spine_terms_rs)
        output = [row[:] for row in grid]
        # Fill spine vertical
        for r in range(min_r, max_r + 1):
            if output[r][spine_c] != 2:
                output[r][spine_c] = 3
        # Connect other terminals horizontally
        for tr, tc in terminals:
            if tc == spine_c:
                continue
            start_c = min(tc, spine_c)
            end_c = max(tc, spine_c)
            for cc in range(start_c, end_c + 1):
                if output[tr][cc] != 2:
                    output[tr][cc] = 3
        return output