def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    strips = []
    for r in range(rows - 1):
        wall_cs = [c for c in range(cols) if grid[r][c] == 8 and grid[r + 1][c] == 8]
        for i in range(len(wall_cs) - 1):
            left = wall_cs[i]
            right = wall_cs[i + 1]
            if right > left + 1:
                start = left + 1
                end = right - 1
                strip_upper = grid[r][start:end + 1]
                strip_lower = grid[r + 1][start:end + 1]
                strips.append([strip_upper, strip_lower])
    if not strips:
        return []
    # Assume all strips have the same width
    width = len(strips[0][0])
    height = 2 * len(strips)
    out_grid = [[0] * width for _ in range(height)]
    cur_row = 0
    for strip in strips:
        out_grid[cur_row] = strip[0][:]
        out_grid[cur_row + 1] = strip[1][:]
        cur_row += 2
    return out_grid