def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    n = 3
    m = 5
    out = [[0] * m for _ in range(m)]
    row_heights = [2, 1, 2]
    row_offsets = [0, 2, 3]
    col_widths = [2, 1, 2]
    col_offsets = [0, 2, 3]
    for i in range(n):
        for j in range(n):
            color = grid_lst[i][j]
            r_start = row_offsets[i]
            r_end = r_start + row_heights[i]
            c_start = col_offsets[j]
            c_end = c_start + col_widths[j]
            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    out[r][c] = color
    return out