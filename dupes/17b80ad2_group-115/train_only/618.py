def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    for j in range(cols):
        if grid[rows - 1][j] == 0:
            continue
        seeds = []
        for i in range(rows):
            if grid[i][j] != 0:
                seeds.append((i, grid[i][j]))
        if not seeds:
            continue
        prev_r = -1
        for k in range(len(seeds)):
            r, c = seeds[k]
            start = prev_r + 1 if k > 0 else 0
            for rr in range(start, r + 1):
                output[rr][j] = c
            prev_r = r
    return output