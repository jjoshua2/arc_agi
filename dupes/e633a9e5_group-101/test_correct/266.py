def transform(grid: list[list[int]]) -> list[list[int]]:
    output = [[0] * 5 for _ in range(5)]
    for i in range(5):
        if i < 2:
            in_row = 0
        elif i == 2:
            in_row = 1
        else:
            in_row = 2
        for j in range(5):
            if j < 2:
                in_col = 0
            elif j == 2:
                in_col = 1
            else:
                in_col = 2
            output[i][j] = grid[in_row][in_col]
    return output