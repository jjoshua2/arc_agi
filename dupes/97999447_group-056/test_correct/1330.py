def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                k = grid[r][c]
                for pos in range(c, cols):
                    if (pos - c) % 2 == 0:
                        output[r][pos] = k
                    else:
                        output[r][pos] = 5
    return output