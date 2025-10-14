def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    swap = {0: 8, 6: 7, 7: 6, 8: 0}

    # Horizontal bars
    left = 0
    right = cols - 1
    for r in range(rows):
        if grid[r][left] == 4 and grid[r][right] == 4:
            for c in range(1, cols - 1):
                val = output[r][c]
                if val in swap:
                    output[r][c] = swap[val]

    # Vertical bars
    for c in range(cols):
        if rows > 1 and grid[0][c] == 4 and grid[rows - 1][c] == 4:
            for r in range(1, rows - 1):
                val = output[r][c]
                if val in swap:
                    output[r][c] = swap[val]

    return output