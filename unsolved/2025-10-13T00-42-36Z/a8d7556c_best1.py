def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    output = [row[:] for row in grid_lst]
    for r in range(rows - 1):
        for c in range(cols - 1):
            if (grid_lst[r][c] == 0 and
                grid_lst[r][c + 1] == 0 and
                grid_lst[r + 1][c] == 0 and
                grid_lst[r + 1][c + 1] == 0):
                output[r][c] = 2
                output[r][c + 1] = 2
                output[r + 1][c] = 2
                output[r + 1][c + 1] = 2
    return output