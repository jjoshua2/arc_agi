def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    m = len(grid_lst)
    if m == 0:
        return []
    n = len(grid_lst[0])
    grid = [row[:] for row in grid_lst]

    # Find number of leading empty rows
    top_empty = 0
    while top_empty < m and all(x == 0 for x in grid[top_empty]):
        top_empty += 1

    # Find number of trailing empty rows
    bottom_empty = 0
    while bottom_empty < m and all(x == 0 for x in grid[m - 1 - bottom_empty]):
        bottom_empty += 1

    colored_start = top_empty
    colored_end = m - bottom_empty
    if colored_start >= colored_end:
        return [row[:] for row in grid]

    # Find blocks: list of (start_r, end_r) exclusive end
    blocks = []
    r = colored_start
    while r < colored_end:
        # Skip empty rows (though shouldn't occur)
        if all(x == 0 for x in grid[r]):
            r += 1
            continue
        # Determine color from first non-zero in this row
        color = None
        for c_val in grid[r]:
            if c_val != 0:
                color = c_val
                break
        if color is None:
            r += 1
            continue
        # Find extent of this color block
        start_r = r
        while r < colored_end:
            row_has_color = False
            row_other_color = False
            for c_val in grid[r]:
                if c_val != 0:
                    if c_val == color:
                        row_has_color = True
                    else:
                        row_other_color = True
            if row_other_color or not row_has_color:
                break
            r += 1
        end_r = r
        blocks.append((start_r, end_r))
        # No r +=1 here, since r is now at end

    # Reverse the blocks
    blocks.reverse()

    # Build new grid
    new_grid = [[0] * n for _ in range(m)]

    # Copy top empty rows
    for i in range(top_empty):
        new_grid[i] = grid[i][:]

    # Place reversed blocks
    current_row = top_empty
    for start_r, end_r in blocks:
        for k in range(start_r, end_r):
            new_grid[current_row] = grid[k][:]
            current_row += 1

    # Copy bottom empty rows (though they are zero)
    for i in range(bottom_empty):
        new_grid[m - bottom_empty + i] = grid[m - bottom_empty + i][:]

    return new_grid