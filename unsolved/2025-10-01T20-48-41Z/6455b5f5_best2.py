def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    output = [row[:] for row in grid_lst]
    grid = grid_lst  # input for checks
    for r in range(rows):
        for h in range(1, rows - r + 1):
            r2 = r + h
            for c1 in range(cols):
                for w in range(1, cols - c1 + 1):
                    c2 = c1 + w
                    # check interior all 0
                    all_zero = True
                    for rr in range(r, r2):
                        for cc in range(c1, c2):
                            if grid[rr][cc] != 0:
                                all_zero = False
                                break
                        if not all_zero:
                            break
                    if not all_zero:
                        continue
                    # check top
                    top_ok = (r == 0)
                    if r > 0:
                        top_ok = all(grid[r - 1][cc] == 2 for cc in range(c1, c2))
                    if not top_ok:
                        continue
                    # check bottom
                    bottom_ok = (r2 == rows)
                    if r2 < rows:
                        bottom_ok = all(grid[r2][cc] == 2 for cc in range(c1, c2))
                    if not bottom_ok:
                        continue
                    # check left
                    left_ok = (c1 == 0)
                    if c1 > 0:
                        left_ok = all(grid[rr][c1 - 1] == 2 for rr in range(r, r2))
                    if not left_ok:
                        continue
                    # check right
                    right_ok = (c2 == cols)
                    if c2 < cols:
                        right_ok = all(grid[rr][c2] == 2 for rr in range(r, r2))
                    if not right_ok:
                        continue
                    # all good, fill
                    color = 8 if h <= 3 else 1
                    for rr in range(r, r2):
                        for cc in range(c1, c2):
                            output[rr][cc] = color
    return output