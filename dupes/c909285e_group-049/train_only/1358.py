def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    for rs in range(rows - 6):
        for cs in range(cols - 6):
            # check top
            top_color = grid[rs][cs]
            top_ok = all(grid[rs][j] == top_color for j in range(cs, cs + 7))
            if not top_ok:
                continue
            # bottom
            bottom_ok = all(grid[rs + 6][j] == top_color for j in range(cs, cs + 7))
            if not bottom_ok:
                continue
            # left
            left_ok = all(grid[i][cs] == top_color for i in range(rs, rs + 7))
            if not left_ok:
                continue
            # right
            right_ok = all(grid[i][cs + 6] == top_color for i in range(rs, rs + 7))
            if not right_ok:
                continue
            # found it
            out = [[grid[i][j] for j in range(cs, cs + 7)] for i in range(rs, rs + 7)]
            return out
    # if none, return empty, but assume there is one
    return []