def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid:
        return []
    h = len(grid)
    w = len(grid[0])
    out_h = 2 * h
    out_w = 2 * w
    out = [[0 for _ in range(out_w)] for _ in range(out_h)]
    for i in range(h):
        for j in range(w):
            c = grid[i][j]
            for di in range(2):
                for dj in range(2):
                    out[2 * i + di][2 * j + dj] = c
    return out