def transform(input_grid: list[list[int]]) -> list[list[int]]:
    if not input_grid or not input_grid[0]:
        return input_grid
    h = len(input_grid)
    w = len(input_grid[0])
    grid = [row[:] for row in input_grid]

    # Find L: the line color
    L = 0
    for r in range(h):
        row_vals = input_grid[r]
        if row_vals and all(x == row_vals[0] and row_vals[0] != 0 for x in row_vals):
            L = row_vals[0]
            break
    if L == 0:
        return grid

    # Find line_rows
    line_rows = []
    for r in range(h):
        if all(grid[r][c] == L for c in range(w)):
            line_rows.append(r)

    # Find line_cols
    line_cols = []
    for c in range(w):
        if all(grid[r][c] == L for r in range(h)):
            line_cols.append(c)

    # section_starts
    section_starts = [0]
    for lr in line_rows:
        ns = lr + 1
        if ns < h:
            section_starts.append(ns)

    # cell_starts
    cell_starts = [0]
    for lc in line_cols:
        ns = lc + 1
        if ns < w:
            cell_starts.append(ns)

    k = len(cell_starts)

    # Patterns for each L (1 where filled)
    patterns = {
        3: [
            [0, 0, 0, 0],
            [1, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 0, 1, 0]
        ],
        8: [
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [1, 1, 1, 1],
            [0, 1, 0, 0]
        ],
        5: [
            [0, 0, 0, 0],
            [1, 0, 1, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 0]
        ]
    }
    pat = patterns.get(L, [[0] * 4 for _ in range(4)])

    # Process each section
    num_sec = len(section_starts)
    for si in range(num_sec):
        s = section_starts[si]
        end = line_rows[si] if si < len(line_rows) else h
        if end - s != 4:
            continue  # Assume exactly 4 rows per section

        # Find C from left cell (cols 0-3, rows s to s+3)
        candidates = set()
        for rr in range(4):
            r = s + rr
            for cc in range(4):
                val = grid[r][cc]
                if 0 < val != L:
                    candidates.add(val)
        if len(candidates) != 1:
            continue
        C = next(iter(candidates))

        # Apply pattern to each vertical cell in the section
        for j in range(k):
            cs = cell_starts[j]
            if cs + 4 > w:
                continue
            for rr in range(4):
                r = s + rr
                for cc in range(4):
                    c = cs + cc
                    if c >= w:
                        continue
                    if pat[rr][cc] == 1:
                        grid[r][c] = C
                    else:
                        grid[r][c] = 0

    return grid