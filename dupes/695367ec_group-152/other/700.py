def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    n = len(grid)
    c = grid[0][0]  # assuming uniform
    step = n + 1
    max_k = 15 // step
    line_pos = [step * k - 1 for k in range(1, max_k + 1)]
    SIZE = 15
    output = [[0] * SIZE for _ in range(SIZE)]
    line_set = set(line_pos)
    for i in range(SIZE):
        for j in range(SIZE):
            if j in line_set:
                output[i][j] = c
        if i in line_set:
            for j in range(SIZE):
                output[i][j] = c
    return output