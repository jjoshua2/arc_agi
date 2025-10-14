def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])
    output = [row[:] for row in grid]
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            # Check horizontal arm: grid[r][c-1], grid[r][c], grid[r][c+1] all 0
            if (grid[r][c - 1] == 0 and
                grid[r][c] == 0 and
                grid[r][c + 1] == 0):
                # Check vertical arm: grid[r-1][c], grid[r][c], grid[r+1][c] all 0
                if (grid[r - 1][c] == 0 and
                    grid[r][c] == 0 and
                    grid[r + 1][c] == 0):
                    # Fill the plus shape
                    output[r][c - 1] = 3
                    output[r][c] = 3
                    output[r][c + 1] = 3
                    output[r - 1][c] = 3
                    output[r + 1][c] = 3
    return output