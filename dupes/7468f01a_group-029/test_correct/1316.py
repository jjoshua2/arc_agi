def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    min_r = rows
    max_r = -1
    min_c = cols
    max_c = -1
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    if max_r < 0:
        return []
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    out = [[0 for _ in range(width)] for _ in range(height)]
    for i in range(height):
        for j in range(width):
            orig_c = min_c + (width - 1 - j)
            out[i][j] = grid[min_r + i][orig_c]
    return out