def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    # Assuming input is always 3x3
    colors = set()
    for row in grid_lst:
        for cell in row:
            colors.add(cell)
    k = len(colors)
    out_h = 3 * k
    out_w = 3 * k
    out = [[0] * out_w for _ in range(out_h)]
    for i in range(3):
        input_row = grid_lst[i]
        out_start_row = i * k
        for out_r in range(k):
            out_row = out[out_start_row + out_r]
            col = 0
            for j in range(3):
                c = input_row[j]
                for _ in range(k):
                    out_row[col] = c
                    col += 1
    return out