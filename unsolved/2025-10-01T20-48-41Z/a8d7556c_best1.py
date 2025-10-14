def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    h = len(grid_lst)
    if h == 0:
        return []
    w = len(grid_lst[0])
    output = [row[:] for row in grid_lst]
    for i in range(h - 1):
        for j in range(w - 1):
            if (grid_lst[i][j] == 0 and
                grid_lst[i][j + 1] == 0 and
                grid_lst[i + 1][j] == 0 and
                grid_lst[i + 1][j + 1] == 0):
                output[i][j] = 2
                output[i][j + 1] = 2
                output[i + 1][j] = 2
                output[i + 1][j + 1] = 2
    return output