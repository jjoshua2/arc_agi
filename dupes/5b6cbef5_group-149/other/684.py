def transform(input_grid: list[list[int]]) -> list[list[int]]:
    if not input_grid or not input_grid[0]:
        return []
    assert len(input_grid) == 4 and len(input_grid[0]) == 4, "Input must be 4x4"
    out_h, out_w = 16, 16
    output = [[0 for _ in range(out_w)] for _ in range(out_h)]
    for i in range(4):
        for j in range(4):
            if input_grid[i][j] != 0:
                for di in range(4):
                    for dj in range(4):
                        output[4 * i + di][4 * j + dj] = input_grid[di][dj]
    return output